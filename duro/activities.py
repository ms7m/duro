"""Temporal activities for running smolagents steps.

Single activity class that constructs a fresh agent per invocation,
restores memory and executor state, runs N steps, and serializes the result.
"""

from __future__ import annotations

import base64
from typing import Any, Callable

from pydantic import BaseModel, Field

from loguru import logger
from smolagents import CodeAgent, MultiStepAgent
from smolagents.agents import ActionOutput
from smolagents.memory import ActionStep, PlanningStep, TaskStep
from smolagents.monitoring import Timing
from temporalio import activity

from duro.serde import (
    deserialize_executor_state,
    deserialize_steps,
    serialize_executor_state,
    serialize_steps,
)
from duro.state_store import InlineStore, PayloadTooLargeError, StateStore


# * Models *


class RunStepsInput(BaseModel):
    """Input to the run_steps activity."""

    task: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    executor_state: str | None = None
    executor_state_ref: str | None = None
    steps_to_run: int = 5
    global_step_number: int = 1


class RunStepsOutput(BaseModel):
    """Output from the run_steps activity."""

    steps: list[dict[str, Any]] = Field(default_factory=list)
    executor_state: str | None = None
    executor_state_ref: str | None = None
    final_answer: Any = None
    done: bool = False
    steps_completed: int = 0
    error: str | None = None
    global_step_number: int = 1


# * Helpers *


def _is_code_agent(agent: MultiStepAgent) -> bool:
    """Detect whether the agent is a CodeAgent with a local executor."""
    return isinstance(agent, CodeAgent) and hasattr(agent, "python_executor")


def _check_local_executor(agent: CodeAgent) -> None:
    """Raise if the agent uses a non-local executor (Docker/E2B/Modal)."""
    from smolagents.local_python_executor import LocalPythonExecutor

    if not isinstance(agent.python_executor, LocalPythonExecutor):
        raise TypeError(
            f"[activities] duro only supports LocalPythonExecutor, "
            f"got {type(agent.python_executor).__name__}. "
            f"Docker/E2B/Modal executors maintain remote state that cannot be serialized."
        )


# * Activity class *


class AgentActivities:
    """Temporal activity that runs smolagents steps."""

    def __init__(
        self,
        agent_factory: Callable[[], MultiStepAgent],
        state_store: StateStore | None = None,
        max_inline_bytes: int = 1_500_000,
    ) -> None:
        self._agent_factory = agent_factory
        self._state_store = state_store or InlineStore(max_inline_bytes=max_inline_bytes)
        self._max_inline_bytes = max_inline_bytes

    @activity.defn
    async def run_steps(self, input: RunStepsInput) -> RunStepsOutput:
        """Run N agent steps, serializing state between cycles."""
        info = activity.info()
        workflow_id = info.workflow_id
        activity_id = info.activity_id

        logger.info(
            f"[activities] Starting run_steps: task={input.task[:80]!r}, "
            f"steps_to_run={input.steps_to_run}, global_step={input.global_step_number}"
        )

        # 1. Build fresh agent
        agent = self._agent_factory()

        # 2. Detect agent type and setup
        is_code = _is_code_agent(agent)
        if is_code:
            assert isinstance(agent, CodeAgent)
            _check_local_executor(agent)
            agent.python_executor.send_tools({**agent.tools, **agent.managed_agents})

        # 3. Restore memory
        if input.steps:
            agent.memory.steps = deserialize_steps(input.steps)
            logger.info(
                f"[activities] Restored {len(agent.memory.steps)} memory steps"
            )

        # 4. Restore executor state
        if is_code:
            assert isinstance(agent, CodeAgent)
            executor_data = await self._resolve_executor_state(input)
            if executor_data:
                deserialize_executor_state(agent.python_executor, executor_data)

        # 5. Ensure TaskStep exists
        has_task_step = any(isinstance(s, TaskStep) for s in agent.memory.steps)
        if not has_task_step:
            agent.memory.steps.insert(0, TaskStep(task=input.task))

        # 6. Step loop
        steps_completed = 0
        final_answer: Any = None
        done = False
        last_error: str | None = None
        global_step = input.global_step_number

        for i in range(input.steps_to_run):
            current_step = global_step + i

            # Planning interval check
            if getattr(agent, "planning_interval", None) is not None:
                should_plan = current_step == 1 or (
                    (current_step - 1) % agent.planning_interval == 0
                )
                if should_plan:
                    logger.info(
                        f"[activities] Generating planning step at global_step={current_step}"
                    )
                    try:
                        for element in agent._generate_planning_step(
                            task=input.task,
                            is_first_step=(current_step == 1),
                            step=current_step,
                        ):
                            if isinstance(element, PlanningStep):
                                agent.memory.steps.append(element)
                    except Exception as e:
                        logger.warning(
                            f"[activities] Planning step failed: {e}"
                        )

            # Create and run action step
            memory_step = ActionStep(
                step_number=current_step,
                timing=Timing(start_time=0.0, end_time=0.0),
            )

            try:
                step_result = agent.step(memory_step)

                # step() returns ActionOutput — _run_stream() normally sets
                # is_final_answer on the memory step, but we bypass _run_stream,
                # so we must check the return value ourselves.
                if isinstance(step_result, ActionOutput) and step_result.is_final_answer:
                    memory_step.is_final_answer = True
                    final_answer = step_result.output
                    done = True

                agent.memory.steps.append(memory_step)
                steps_completed += 1

                if done:
                    logger.info(
                        f"[activities] Agent finished at step {current_step}"
                    )
                    break

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[activities] Step {current_step} raised: {e}"
                )
                # Append step anyway — agent may self-correct
                if memory_step.error is None:
                    memory_step.error = e
                agent.memory.steps.append(memory_step)
                steps_completed += 1

        # 7. Serialize output
        serialized_steps = serialize_steps(agent.memory.steps)
        executor_state_inline: str | None = None
        executor_state_ref: str | None = None

        if is_code:
            assert isinstance(agent, CodeAgent)
            state_result = serialize_executor_state(agent.python_executor)

            if state_result.data:
                if state_result.size_bytes <= self._max_inline_bytes:
                    executor_state_inline = state_result.data
                else:
                    # Offload to blob store
                    key = f"{workflow_id}/{activity_id}/executor_state"
                    raw_bytes = base64.b64decode(state_result.data)
                    try:
                        await self._state_store.put(key, raw_bytes)
                        executor_state_ref = key
                        logger.info(
                            f"[activities] Offloaded executor state "
                            f"({state_result.size_bytes:,} bytes) to store: {key}"
                        )
                    except PayloadTooLargeError:
                        raise
                    except Exception as e:
                        logger.error(
                            f"[activities] Failed to offload executor state: {e}"
                        )
                        raise

        new_global_step = global_step + steps_completed

        logger.info(
            f"[activities] Completed: {steps_completed} steps, "
            f"done={done}, global_step={new_global_step}"
        )

        return RunStepsOutput(
            steps=serialized_steps,
            executor_state=executor_state_inline,
            executor_state_ref=executor_state_ref,
            final_answer=final_answer,
            done=done,
            steps_completed=steps_completed,
            error=last_error,
            global_step_number=new_global_step,
        )

    async def _resolve_executor_state(self, input: RunStepsInput) -> str | None:
        """Resolve executor state from inline data or blob store reference."""
        if input.executor_state:
            return input.executor_state
        if input.executor_state_ref:
            logger.info(
                f"[activities] Fetching executor state from store: {input.executor_state_ref}"
            )
            raw_bytes = await self._state_store.get(input.executor_state_ref)
            return base64.b64encode(raw_bytes).decode()
        return None
