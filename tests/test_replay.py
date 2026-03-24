"""Integration tests proving duro's killer feature: crash recovery without re-invoking the LLM.

These tests use Temporal's test server to demonstrate that:
1. When a workflow resumes after a crash, completed activities are NOT re-executed
2. Memory and executor state survive across cycle boundaries
3. Step N can continue from where step N-1 left off without replaying steps 1..N-1

The mock agent tracks every LLM call via a shared counter, so we can assert
exactly how many times the model was invoked.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest
from smolagents.agents import ActionOutput
from smolagents.memory import ActionStep, TaskStep
from smolagents.models import ChatMessage
from smolagents.monitoring import Timing
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from duro.activities import AgentActivities, RunStepsInput, RunStepsOutput
from duro.serde import deserialize_steps, serialize_steps
from duro.workflows import AgentWorkflow, AgentWorkflowParams


# * Shared call counter — tracks LLM invocations across agent instances *


class CallTracker:
    """Thread-safe counter for tracking how many times agent.step() is called."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._step_log: list[int] = []

    def record(self, step_number: int) -> None:
        with self._lock:
            self._count += 1
            self._step_log.append(step_number)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def step_log(self) -> list[int]:
        with self._lock:
            return list(self._step_log)

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._step_log.clear()


# * Mock agent — simulates an LLM agent without real model calls *


class MockAgent:
    """Fake agent that counts step() calls and finishes after N steps.

    Every call to step() represents what would be an LLM call in production.
    The tracker lets us assert across agent instances (since the factory
    creates a new agent each cycle).
    """

    def __init__(
        self,
        tracker: CallTracker,
        finish_at_step: int = 8,
        fail_at_step: int | None = None,
    ) -> None:
        self.tracker = tracker
        self.finish_at_step = finish_at_step
        self.fail_at_step = fail_at_step
        self.memory = MagicMock()
        self.memory.steps = []
        self.tools: dict[str, Any] = {}
        self.managed_agents: dict[str, Any] = {}
        self.planning_interval = None

    def step(self, memory_step: ActionStep) -> Any:
        step_num = memory_step.step_number
        self.tracker.record(step_num)

        # Simulate a crash at a specific step
        if self.fail_at_step is not None and step_num == self.fail_at_step:
            raise RuntimeError(f"Simulated crash at step {step_num}")

        # Simulate LLM output
        memory_step.model_output = f"Thinking about step {step_num}..."
        memory_step.model_output_message = ChatMessage(
            role="assistant",
            content=f"Step {step_num} output",
        )
        memory_step.observations = f"Result of step {step_num}"

        if step_num >= self.finish_at_step:
            return ActionOutput(
                output=f"Final answer after {step_num} steps",
                is_final_answer=True,
            )

        return ActionOutput(output=None, is_final_answer=False)


# * Test: Temporal replay does NOT re-invoke the LLM *


@pytest.mark.asyncio
async def test_completed_cycles_are_not_reexecuted() -> None:
    """Prove that when a workflow runs multiple cycles, completed cycles
    are replayed from Temporal history — the LLM is never re-called.

    Setup:
    - Agent finishes at step 8
    - 3 steps per cycle → cycle 1 (steps 1-3), cycle 2 (steps 4-6), cycle 3 (steps 7-8)
    - Total LLM calls should be exactly 8

    If Temporal were re-executing activities on replay, we'd see >8 calls.
    """
    tracker = CallTracker()

    def make_agent() -> MockAgent:
        return MockAgent(tracker=tracker, finish_at_step=8)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        activities = AgentActivities(agent_factory=make_agent)

        async with Worker(
            env.client,
            task_queue="test-replay",
            workflows=[AgentWorkflow],
            activities=[activities.run_steps],
        ):
            result = await env.client.execute_workflow(
                AgentWorkflow.run,
                AgentWorkflowParams(
                    task="Count to 8",
                    steps_per_cycle=3,
                    max_cycles=10,
                ),
                id="test-no-replay-calls",
                task_queue="test-replay",
            )

        assert result.answer == "Final answer after 8 steps"
        assert not result.timed_out

        # The killer assertion: exactly 8 LLM calls, not more
        assert tracker.count == 8, (
            f"Expected exactly 8 LLM calls, got {tracker.count}. "
            f"Steps called: {tracker.step_log}"
        )
        # Every step 1-8 was called exactly once
        assert tracker.step_log == [1, 2, 3, 4, 5, 6, 7, 8]


# * Test: State survives across cycle boundaries *


@pytest.mark.asyncio
async def test_state_carries_across_cycles() -> None:
    """Prove that memory steps from cycle 1 are visible in cycle 2.

    The mock agent inspects its own memory to verify that previous steps
    were restored before the next cycle runs.
    """
    steps_seen_at_cycle_start: list[int] = []
    tracker = CallTracker()

    class StateCheckingAgent(MockAgent):
        def step(self, memory_step: ActionStep) -> Any:
            # On each step, record how many prior steps are in memory
            if memory_step.step_number == 1 or (memory_step.step_number - 1) % 2 == 0:
                # First step of each cycle — check restored memory
                prior_action_steps = [
                    s for s in self.memory.steps if isinstance(s, ActionStep)
                ]
                steps_seen_at_cycle_start.append(len(prior_action_steps))

            return super().step(memory_step)

    def make_agent() -> StateCheckingAgent:
        return StateCheckingAgent(tracker=tracker, finish_at_step=6)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        activities = AgentActivities(agent_factory=make_agent)

        async with Worker(
            env.client,
            task_queue="test-state",
            workflows=[AgentWorkflow],
            activities=[activities.run_steps],
        ):
            result = await env.client.execute_workflow(
                AgentWorkflow.run,
                AgentWorkflowParams(
                    task="Test state carry",
                    steps_per_cycle=2,
                    max_cycles=10,
                ),
                id="test-state-carry",
                task_queue="test-state",
            )

        assert result.answer == "Final answer after 6 steps"
        assert tracker.count == 6

        # Cycle 1 starts with 0 prior steps (fresh)
        # Cycle 2 starts with 2 prior action steps (from cycle 1)
        # Cycle 3 starts with 4 prior action steps (from cycles 1+2)
        assert steps_seen_at_cycle_start[0] == 0, "Cycle 1 should start fresh"
        assert steps_seen_at_cycle_start[1] == 2, "Cycle 2 should see 2 prior steps"
        assert steps_seen_at_cycle_start[2] == 4, "Cycle 3 should see 4 prior steps"


# * Test: Crash at step N, recovery continues from N (not from 1) *


@pytest.mark.asyncio
async def test_crash_recovery_continues_from_last_checkpoint() -> None:
    """Prove that if a worker crashes mid-cycle, Temporal retries ONLY
    the failed activity — steps from completed cycles are NOT re-run.

    Setup:
    - 3 steps per cycle
    - Cycle 1: steps 1-3 complete successfully (3 LLM calls)
    - Cycle 2: agent_factory crashes (simulating worker death) → activity fails
    - Temporal retries cycle 2: factory succeeds, steps 4-6 run (3 LLM calls)
    - Cycle 3: steps 7-8, agent finishes (2 LLM calls)

    Key assertion: steps 1-3 from cycle 1 are NEVER re-executed.
    Temporal replays cycle 1's recorded output and only re-runs cycle 2.
    """
    tracker = CallTracker()
    factory_call_count = 0

    def make_agent() -> MockAgent:
        nonlocal factory_call_count
        factory_call_count += 1

        # factory_call_count:
        #   1 = cycle 1 (steps 1-3, succeeds)
        #   2 = cycle 2, first attempt (crash before any steps run)
        #   3 = cycle 2, retry (succeeds, steps 4-6)
        #   4 = cycle 3 (steps 7-8, finishes)
        if factory_call_count == 2:
            raise RuntimeError("Simulated worker crash during cycle 2")

        return MockAgent(tracker=tracker, finish_at_step=8)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        activities = AgentActivities(agent_factory=make_agent)

        async with Worker(
            env.client,
            task_queue="test-crash",
            workflows=[AgentWorkflow],
            activities=[activities.run_steps],
        ):
            result = await env.client.execute_workflow(
                AgentWorkflow.run,
                AgentWorkflowParams(
                    task="Crash and recover",
                    steps_per_cycle=3,
                    max_cycles=10,
                    activity_timeout_seconds=10,
                ),
                id="test-crash-recovery",
                task_queue="test-crash",
            )

        assert result.answer == "Final answer after 8 steps"

        # Steps 1-3 were called exactly once (cycle 1, never replayed)
        cycle1_calls = [s for s in tracker.step_log if s <= 3]
        assert cycle1_calls == [1, 2, 3], (
            f"Cycle 1 steps should run exactly once, got: {cycle1_calls}"
        )

        # Total: 3 (cycle1) + 0 (crashed cycle2) + 3 (retried cycle2) + 2 (cycle3)
        # = 8 LLM calls. NOT re-running steps 1-3 on retry.
        assert tracker.count == 8, (
            f"Expected 8 total calls (no re-execution), got {tracker.count}. "
            f"Log: {tracker.step_log}"
        )
        assert tracker.step_log == [1, 2, 3, 4, 5, 6, 7, 8]


# * Test: Direct serde round-trip proves memory fidelity (no Temporal needed) *


@pytest.mark.asyncio
async def test_manual_serialize_restore_no_llm_needed() -> None:
    """Prove the serialize/restore cycle preserves all state the LLM needs.

    Run 3 steps, serialize memory, create a brand new agent, restore memory,
    verify the new agent sees all previous steps — demonstrating that
    recovery doesn't require re-running the LLM.
    """
    tracker = CallTracker()

    # Phase 1: Run 3 steps
    agent1 = MockAgent(tracker=tracker, finish_at_step=10)
    agent1.memory.steps = [TaskStep(task="Test task")]

    for i in range(1, 4):
        step = ActionStep(
            step_number=i,
            timing=Timing(start_time=0.0, end_time=0.0),
        )
        agent1.step(step)
        agent1.memory.steps.append(step)

    assert tracker.count == 3

    # Serialize state
    serialized = serialize_steps(agent1.memory.steps)

    # Phase 2: New agent, restore state, continue
    tracker2 = CallTracker()
    agent2 = MockAgent(tracker=tracker2, finish_at_step=5)
    agent2.memory.steps = deserialize_steps(serialized)

    # Verify restored memory has all 4 items (1 TaskStep + 3 ActionSteps)
    assert len(agent2.memory.steps) == 4
    assert isinstance(agent2.memory.steps[0], TaskStep)
    assert all(isinstance(s, ActionStep) for s in agent2.memory.steps[1:])

    # Continue from step 4 — no need to re-run steps 1-3
    for i in range(4, 6):
        step = ActionStep(
            step_number=i,
            timing=Timing(start_time=0.0, end_time=0.0),
        )
        agent2.step(step)
        agent2.memory.steps.append(step)

    # Agent2 only made 2 LLM calls (steps 4-5), not 5
    assert tracker2.count == 2
    # Original agent's calls are untouched
    assert tracker.count == 3
    # Total memory has all 6 items
    assert len(agent2.memory.steps) == 6
