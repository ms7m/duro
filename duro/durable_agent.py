"""Developer-facing wrapper that hides Temporal internals.

Provides DurableAgent with run() and create_worker() methods.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from loguru import logger
from smolagents import MultiStepAgent
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from duro.activities import AgentActivities
from duro.state_store import StateStore
from duro.workflows import AgentWorkflow, AgentWorkflowParams


class DurableAgent:
    """High-level wrapper that runs a smolagents agent as a Temporal workflow."""

    def __init__(
        self,
        agent_factory: Callable[[], MultiStepAgent],
        task_queue: str = "smolagents",
        max_cycles: int = 10,
        steps_per_cycle: int = 5,
        activity_timeout_seconds: int = 300,
        state_store: StateStore | None = None,
        max_inline_bytes: int = 1_500_000,
    ) -> None:
        self._agent_factory = agent_factory
        self._task_queue = task_queue
        self._max_cycles = max_cycles
        self._steps_per_cycle = steps_per_cycle
        self._activity_timeout_seconds = activity_timeout_seconds
        self._state_store = state_store
        self._max_inline_bytes = max_inline_bytes

    async def run(
        self,
        task: str,
        temporal_client: Client | None = None,
        workflow_id: str | None = None,
    ) -> Any:
        """Run the agent as a Temporal workflow. Returns the final answer."""
        client = temporal_client or await Client.connect(
            "localhost:7233",
            data_converter=pydantic_data_converter,
        )
        wf_id = workflow_id or f"duro-{uuid.uuid4().hex[:12]}"

        logger.info(
            f"[durable_agent] Starting workflow {wf_id}: {task[:80]!r}"
        )

        result = await client.execute_workflow(
            AgentWorkflow.run,
            AgentWorkflowParams(
                task=task,
                max_cycles=self._max_cycles,
                steps_per_cycle=self._steps_per_cycle,
                activity_timeout_seconds=self._activity_timeout_seconds,
            ),
            id=wf_id,
            task_queue=self._task_queue,
        )

        if result.timed_out:
            logger.warning(
                f"[durable_agent] Workflow {wf_id} timed out after "
                f"{result.total_steps} steps"
            )
        else:
            logger.info(
                f"[durable_agent] Workflow {wf_id} completed in "
                f"{result.total_steps} steps"
            )

        return result.answer

    def create_worker(self, client: Client) -> Worker:
        """Create a Temporal worker pre-configured for this agent."""
        activities = AgentActivities(
            agent_factory=self._agent_factory,
            state_store=self._state_store,
            max_inline_bytes=self._max_inline_bytes,
        )

        return Worker(
            client,
            task_queue=self._task_queue,
            workflows=[AgentWorkflow],
            activities=[activities.run_steps],
        )
