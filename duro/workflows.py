"""Temporal workflow for orchestrating agent execution cycles.

Deliberately simple — the complexity lives in the activity.
The workflow is deterministic: it loops calling the run_steps activity
and carrying forward serialized state between cycles.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pydantic import BaseModel

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from loguru import logger

    from duro.activities import AgentActivities, RunStepsInput, RunStepsOutput


# * Models *


class AgentWorkflowParams(BaseModel):
    """Parameters for starting an AgentWorkflow."""

    task: str
    max_cycles: int = 10
    steps_per_cycle: int = 5
    activity_timeout_seconds: int = 300


class AgentWorkflowResult(BaseModel):
    """Result returned by AgentWorkflow on completion."""

    answer: Any = None
    timed_out: bool = False
    total_steps: int = 0
    error: str | None = None


# * Workflow *


@workflow.defn(sandboxed=False)
class AgentWorkflow:
    """Temporal workflow that runs a smolagents agent in durable cycles."""

    @workflow.run
    async def run(self, params: AgentWorkflowParams) -> AgentWorkflowResult:
        logger.info(
            f"[workflow] Starting: task={params.task[:80]!r}, "
            f"max_cycles={params.max_cycles}, steps_per_cycle={params.steps_per_cycle}"
        )

        steps: list[dict[str, Any]] = []
        executor_state: str | None = None
        executor_state_ref: str | None = None
        global_step_number: int = 1
        total_steps: int = 0
        last_error: str | None = None

        for cycle in range(params.max_cycles):
            logger.info(
                f"[workflow] Cycle {cycle + 1}/{params.max_cycles}, "
                f"global_step={global_step_number}"
            )

            output: RunStepsOutput = await workflow.execute_activity_method(
                AgentActivities.run_steps,
                RunStepsInput(
                    task=params.task,
                    steps=steps,
                    executor_state=executor_state,
                    executor_state_ref=executor_state_ref,
                    steps_to_run=params.steps_per_cycle,
                    global_step_number=global_step_number,
                ),
                start_to_close_timeout=timedelta(
                    seconds=params.activity_timeout_seconds
                ),
            )

            steps = output.steps
            executor_state = output.executor_state
            executor_state_ref = output.executor_state_ref
            global_step_number = output.global_step_number
            total_steps += output.steps_completed
            last_error = output.error

            if output.done:
                logger.info(
                    f"[workflow] Agent finished after {cycle + 1} cycles, "
                    f"{total_steps} total steps"
                )
                return AgentWorkflowResult(
                    answer=output.final_answer,
                    total_steps=total_steps,
                )

        logger.warning(
            f"[workflow] Agent timed out after {params.max_cycles} cycles, "
            f"{total_steps} total steps"
        )
        return AgentWorkflowResult(
            answer=None,
            timed_out=True,
            total_steps=total_steps,
            error=last_error,
        )
