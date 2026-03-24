"""Unit tests for duro.serde — no LLM, no Temporal required."""

from __future__ import annotations

from smolagents.local_python_executor import LocalPythonExecutor
from smolagents.memory import ActionStep, PlanningStep, TaskStep, ToolCall
from smolagents.models import ChatMessage
from smolagents.monitoring import Timing

from duro.serde import (
    _TYPE_HANDLERS,
    _type_key,
    deserialize_executor_state,
    deserialize_step,
    deserialize_steps,
    register_type_handler,
    serialize_executor_state,
    serialize_step,
    serialize_steps,
)

_DUMMY_TIMING = Timing(start_time=0.0, end_time=0.0)


def _make_executor() -> LocalPythonExecutor:
    return LocalPythonExecutor(additional_authorized_imports=["pandas", "numpy"])


# * Memory step round-trip tests *


class TestTaskStepRoundTrip:
    def test_basic(self) -> None:
        step = TaskStep(task="Analyze this dataset")
        data = serialize_step(step)
        restored = deserialize_step(data)

        assert isinstance(restored, TaskStep)
        assert restored.task == "Analyze this dataset"

    def test_to_messages_match(self) -> None:
        step = TaskStep(task="Do something")
        restored = deserialize_step(serialize_step(step))

        original_msgs = step.to_messages()
        restored_msgs = restored.to_messages()
        assert original_msgs == restored_msgs


class TestActionStepRoundTrip:
    def _make_action_step(self, **overrides: object) -> ActionStep:
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=100.0, end_time=200.0),
            model_output="I'll use pandas to analyze.",
            model_output_message=ChatMessage(
                role="assistant", content="I'll use pandas to analyze."
            ),
            observations="DataFrame with 10 rows",
            tool_calls=[ToolCall(name="python_interpreter", arguments="import pandas", id="tc1")],
            is_final_answer=False,
            code_action="import pandas as pd",
            action_output="None",
        )
        for k, v in overrides.items():
            setattr(step, k, v)
        return step

    def test_basic(self) -> None:
        step = self._make_action_step()
        data = serialize_step(step)
        restored = deserialize_step(data)

        assert isinstance(restored, ActionStep)
        assert restored.step_number == 1
        assert restored.model_output == "I'll use pandas to analyze."
        assert restored.observations == "DataFrame with 10 rows"
        assert restored.is_final_answer is False

    def test_to_messages_match(self) -> None:
        step = self._make_action_step()
        restored = deserialize_step(serialize_step(step))

        original_msgs = step.to_messages()
        restored_msgs = restored.to_messages()
        assert original_msgs == restored_msgs

    def test_with_error(self) -> None:
        error = Exception("Something went wrong")
        step = self._make_action_step(error=error)
        data = serialize_step(step)
        restored = deserialize_step(data)

        assert restored.error is not None
        assert "Something went wrong" in str(restored.error)

    def test_final_answer(self) -> None:
        step = self._make_action_step(is_final_answer=True)
        data = serialize_step(step)
        restored = deserialize_step(data)
        assert restored.is_final_answer is True

    def test_none_observations(self) -> None:
        step = self._make_action_step(observations=None, model_output=None)
        data = serialize_step(step)
        restored = deserialize_step(data)
        assert restored.observations is None


class TestPlanningStepRoundTrip:
    def test_basic(self) -> None:
        step = PlanningStep(
            model_input_messages=[{"role": "user", "content": "Plan this"}],
            model_output_message=ChatMessage(
                role="assistant", content="Here is my plan"
            ),
            plan="1. First step\n2. Second step",
            timing=_DUMMY_TIMING,
        )

        data = serialize_step(step)
        restored = deserialize_step(data)

        assert isinstance(restored, PlanningStep)
        assert restored.plan == "1. First step\n2. Second step"
        assert restored.model_output_message.content == "Here is my plan"

    def test_to_messages_match(self) -> None:
        step = PlanningStep(
            model_input_messages=[],
            model_output_message=ChatMessage(
                role="assistant", content="My plan"
            ),
            plan="Do stuff",
            timing=_DUMMY_TIMING,
        )
        restored = deserialize_step(serialize_step(step))

        original_msgs = step.to_messages()
        restored_msgs = restored.to_messages()
        assert original_msgs == restored_msgs


class TestStepsListRoundTrip:
    def test_mixed_steps(self) -> None:
        steps = [
            TaskStep(task="Do something"),
            ActionStep(step_number=1, timing=_DUMMY_TIMING),
            PlanningStep(
                model_input_messages=[],
                model_output_message=ChatMessage(role="assistant", content="Plan"),
                plan="plan text",
                timing=_DUMMY_TIMING,
            ),
        ]
        serialized = serialize_steps(steps)
        restored = deserialize_steps(serialized)

        assert len(restored) == 3
        assert isinstance(restored[0], TaskStep)
        assert isinstance(restored[1], ActionStep)
        assert isinstance(restored[2], PlanningStep)


# * Executor state tests *


class TestExecutorStateRoundTrip:
    def test_empty_state(self) -> None:
        executor = _make_executor()
        result = serialize_executor_state(executor)
        assert result.data is None
        assert result.size_bytes == 0

    def test_basic_variables(self) -> None:
        executor = _make_executor()
        executor.state["x"] = 42
        executor.state["name"] = "test"
        executor.state["data"] = [1, 2, 3]

        result = serialize_executor_state(executor)
        assert result.data is not None
        assert result.size_bytes > 0

        new_executor = _make_executor()
        deserialize_executor_state(new_executor, result.data)

        assert new_executor.state["x"] == 42
        assert new_executor.state["name"] == "test"
        assert new_executor.state["data"] == [1, 2, 3]

    def test_with_modules(self) -> None:
        import json

        executor = _make_executor()
        executor.state["json"] = json
        executor.state["result"] = 99

        result = serialize_executor_state(executor)
        assert result.data is not None

        new_executor = _make_executor()
        deserialize_executor_state(new_executor, result.data)
        assert new_executor.state["json"] is json
        assert new_executor.state["result"] == 99

    def test_internal_keys_skipped(self) -> None:
        executor = _make_executor()
        executor.state["__name__"] = "__main__"
        executor.state["_print_outputs"] = "something"
        executor.state["_operations_count"] = {}
        executor.state["user_var"] = "keep me"

        result = serialize_executor_state(executor)
        new_executor = _make_executor()
        deserialize_executor_state(new_executor, result.data)

        assert "user_var" in new_executor.state
        assert new_executor.state.get("__name__") == "__main__"

    def test_graceful_degradation(self) -> None:
        """Unpicklable values are dropped; picklable values survive."""

        class Unpicklable:
            def __reduce__(self) -> None:
                raise TypeError("Can't pickle me")

        executor = _make_executor()
        executor.state["good"] = 42
        executor.state["bad"] = Unpicklable()

        result = serialize_executor_state(executor)
        new_executor = _make_executor()
        deserialize_executor_state(new_executor, result.data)

        assert new_executor.state["good"] == 42
        assert "bad" not in new_executor.state

    def test_none_input(self) -> None:
        executor = _make_executor()
        deserialize_executor_state(executor, None)

    def test_empty_string_input(self) -> None:
        executor = _make_executor()
        deserialize_executor_state(executor, "")


# * Type handler registry tests *


class TestTypeHandlerRegistry:
    def test_register_custom_handler(self) -> None:
        class MyType:
            def __init__(self, value: int) -> None:
                self.value = value

            def to_json(self) -> str:
                return str(self.value)

            @classmethod
            def from_json(cls, data: str) -> MyType:
                return cls(int(data))

        register_type_handler(
            MyType,
            serialize_fn=lambda v: v.to_json(),
            deserialize_fn=lambda b: MyType.from_json(b),
        )

        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.state["obj"] = MyType(42)

        result = serialize_executor_state(executor)
        new_executor = LocalPythonExecutor(additional_authorized_imports=[])
        deserialize_executor_state(new_executor, result.data)

        assert new_executor.state["obj"].value == 42

        # Cleanup
        del _TYPE_HANDLERS[_type_key(MyType)]
