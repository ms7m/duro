"""Serialization layer for smolagents memory steps and executor state.

Handles round-trip conversion of smolagents internals to/from JSON-safe dicts.
Design principle: only serialize fields that to_messages() reads.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Callable

from pydantic import BaseModel

import dill
from loguru import logger
from smolagents.local_python_executor import LocalPythonExecutor
from smolagents.memory import ActionStep, PlanningStep, TaskStep, ToolCall
from smolagents.models import ChatMessage
from smolagents.monitoring import Timing

# * Type handler registry *

_TypeHandler = tuple[Callable[[Any], str], Callable[[str], Any]]
_TYPE_HANDLERS: dict[str, _TypeHandler] = {}


def _type_key(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def register_type_handler(
    cls: type,
    serialize_fn: Callable[[Any], str],
    deserialize_fn: Callable[[str], Any],
) -> None:
    """Register a custom serializer/deserializer for a type that dill can't handle."""
    _TYPE_HANDLERS[_type_key(cls)] = (serialize_fn, deserialize_fn)


def _register_builtin_handlers() -> None:
    """Register handlers for known-problematic types (docx, pptx)."""
    try:
        import docx  # type: ignore[import-untyped]

        def _save_to_bytes(obj: Any) -> bytes:
            buf = io.BytesIO()
            obj.save(buf)
            return buf.getvalue()

        register_type_handler(
            docx.Document,
            serialize_fn=lambda v: base64.b64encode(_save_to_bytes(v)).decode(),
            deserialize_fn=lambda b: docx.Document(io.BytesIO(base64.b64decode(b))),
        )
    except ImportError:
        pass

    try:
        import pptx  # type: ignore[import-untyped]

        def _save_pptx_bytes(obj: Any) -> bytes:
            buf = io.BytesIO()
            obj.save(buf)
            return buf.getvalue()

        register_type_handler(
            pptx.Presentation,
            serialize_fn=lambda v: base64.b64encode(_save_pptx_bytes(v)).decode(),
            deserialize_fn=lambda b: pptx.Presentation(io.BytesIO(base64.b64decode(b))),
        )
    except ImportError:
        pass


_register_builtin_handlers()

# * Internal keys to skip when serializing executor state *

_EXECUTOR_INTERNAL_KEYS = frozenset({
    "__name__",
    "__builtins__",
    "_print_outputs",
    "_operations_count",
})

# * Memory step serialization *


def _serialize_chat_message(msg: ChatMessage) -> dict[str, Any]:
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": msg.tool_calls,
        "raw": msg.raw,
    }


def _deserialize_chat_message(data: dict[str, Any]) -> ChatMessage:
    return ChatMessage(
        role=data["role"],
        content=data.get("content"),
        tool_calls=data.get("tool_calls"),
        raw=data.get("raw"),
    )


def _safe_str(value: Any) -> str | None:
    """Best-effort string conversion for non-JSON-safe values."""
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def serialize_step(step: TaskStep | ActionStep | PlanningStep) -> dict[str, Any]:
    """Serialize a single memory step to a JSON-safe dict."""
    if isinstance(step, TaskStep):
        return {
            "type": "TaskStep",
            "task": step.task,
        }

    if isinstance(step, ActionStep):
        data: dict[str, Any] = {
            "type": "ActionStep",
            "step_number": step.step_number,
            "model_output": step.model_output,
            "observations": step.observations,
            "is_final_answer": step.is_final_answer,
        }
        if step.model_output_message is not None:
            data["model_output_message"] = _serialize_chat_message(
                step.model_output_message
            )
        if step.tool_calls is not None:
            data["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments, "id": tc.id}
                if isinstance(tc, ToolCall)
                else tc
                for tc in step.tool_calls
            ]
        if step.error is not None:
            data["error"] = _serialize_error(step.error)
        if step.code_action is not None:
            data["code_action"] = step.code_action
        if step.action_output is not None:
            data["action_output"] = _safe_str(step.action_output)
        if step.token_usage is not None:
            data["token_usage"] = step.token_usage
        return data

    if isinstance(step, PlanningStep):
        data = {
            "type": "PlanningStep",
            "plan": step.plan,
        }
        if step.model_output_message is not None:
            data["model_output_message"] = _serialize_chat_message(
                step.model_output_message
            )
        if step.token_usage is not None:
            data["token_usage"] = step.token_usage
        return data

    raise TypeError(f"[serde] Unknown step type: {type(step)}")


def _serialize_error(error: Any) -> dict[str, Any] | None:
    """Serialize an error field — may be AgentError, Exception, or dict."""
    if error is None:
        return None
    if isinstance(error, dict):
        return error
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def _deserialize_error(data: dict[str, Any] | None) -> Exception | None:
    """Deserialize an error field back to an Exception."""
    if data is None:
        return None
    return Exception(data.get("message", str(data)))


def deserialize_step(data: dict[str, Any]) -> TaskStep | ActionStep | PlanningStep:
    """Deserialize a dict back to a memory step."""
    step_type = data["type"]

    if step_type == "TaskStep":
        step = TaskStep(task=data["task"])
        step.task_images = None
        return step

    if step_type == "ActionStep":
        dummy_timing = Timing(start_time=0.0, end_time=0.0)
        step = ActionStep(step_number=data.get("step_number", 0), timing=dummy_timing)
        step.model_output = data.get("model_output")
        step.observations = data.get("observations")
        step.observations_images = None
        step.is_final_answer = data.get("is_final_answer", False)
        raw_tcs = data.get("tool_calls")
        if raw_tcs is not None:
            step.tool_calls = [
                ToolCall(name=tc["name"], arguments=tc["arguments"], id=tc["id"])
                if isinstance(tc, dict)
                else tc
                for tc in raw_tcs
            ]
        else:
            step.tool_calls = None
        step.error = _deserialize_error(data.get("error"))
        step.code_action = data.get("code_action")
        step.action_output = data.get("action_output")
        step.token_usage = data.get("token_usage")
        if "model_output_message" in data and data["model_output_message"] is not None:
            step.model_output_message = _deserialize_chat_message(
                data["model_output_message"]
            )
        else:
            step.model_output_message = None
        return step

    if step_type == "PlanningStep":
        msg = None
        if "model_output_message" in data and data["model_output_message"] is not None:
            msg = _deserialize_chat_message(data["model_output_message"])
        dummy_timing = Timing(start_time=0.0, end_time=0.0)
        step = PlanningStep(
            model_input_messages=[],
            model_output_message=msg or ChatMessage(role="assistant", content=""),
            plan=data.get("plan", ""),
            timing=dummy_timing,
        )
        step.token_usage = data.get("token_usage")
        return step

    raise ValueError(f"[serde] Unknown step type: {step_type}")


def serialize_steps(
    steps: list[TaskStep | ActionStep | PlanningStep],
) -> list[dict[str, Any]]:
    """Serialize a list of memory steps."""
    return [serialize_step(s) for s in steps]


def deserialize_steps(
    data: list[dict[str, Any]],
) -> list[TaskStep | ActionStep | PlanningStep]:
    """Deserialize a list of memory step dicts."""
    return [deserialize_step(d) for d in data]


# * Executor state serialization *


class ExecutorStateResult(BaseModel):
    """Result of executor state serialization."""

    data: str | None = None  # base64-encoded bytes, or None if empty
    size_bytes: int = 0  # raw size before base64


def serialize_executor_state(executor: LocalPythonExecutor) -> ExecutorStateResult:
    """Serialize executor.state, handling type handlers and graceful degradation."""
    user_state: dict[str, Any] = {}
    handled_state: dict[str, dict[str, str]] = {}

    for key, value in executor.state.items():
        if key in _EXECUTOR_INTERNAL_KEYS:
            continue
        if key.startswith("_"):
            continue

        type_k = _type_key(type(value))
        if type_k in _TYPE_HANDLERS:
            try:
                ser_fn, _ = _TYPE_HANDLERS[type_k]
                handled_state[key] = {
                    "type_key": type_k,
                    "data": ser_fn(value),
                }
                logger.debug(f"[serde] Type handler serialized {key} ({type_k})")
            except Exception as e:
                logger.warning(
                    f"[serde] Type handler failed for {key} ({type_k}): {e} — dropping variable"
                )
            continue

        user_state[key] = value

    if not user_state and not handled_state:
        return ExecutorStateResult(data=None, size_bytes=0)

    # Try dill on the remaining user state, dropping individual failures
    safe_state: dict[str, Any] = {}
    for key, value in user_state.items():
        try:
            dill.dumps(value)
            safe_state[key] = value
        except Exception as e:
            logger.warning(
                f"[serde] Cannot serialize variable '{key}' ({type(value).__name__}): {e} — dropping"
            )

    payload = {
        "dill_state": base64.b64encode(dill.dumps(safe_state)).decode()
        if safe_state
        else None,
        "handled_state": handled_state if handled_state else None,
    }

    raw_bytes = dill.dumps(payload)
    encoded = base64.b64encode(raw_bytes).decode()
    size = len(raw_bytes)

    if size > 1_000_000:
        logger.warning(
            f"[serde] Executor state is {size:,} bytes — approaching Temporal payload limits"
        )

    return ExecutorStateResult(data=encoded, size_bytes=size)


def deserialize_executor_state(
    executor: LocalPythonExecutor,
    encoded: str | None,
) -> None:
    """Restore executor state from serialized data."""
    if not encoded:
        return

    raw_bytes = base64.b64decode(encoded)
    payload: dict[str, Any] = dill.loads(raw_bytes)

    restored: dict[str, Any] = {}

    # Restore dill-serialized state
    if payload.get("dill_state"):
        dill_bytes = base64.b64decode(payload["dill_state"])
        restored.update(dill.loads(dill_bytes))

    # Restore type-handled state
    if payload.get("handled_state"):
        for key, handler_data in payload["handled_state"].items():
            type_k = handler_data["type_key"]
            if type_k in _TYPE_HANDLERS:
                try:
                    _, deser_fn = _TYPE_HANDLERS[type_k]
                    restored[key] = deser_fn(handler_data["data"])
                    logger.debug(
                        f"[serde] Type handler restored {key} ({type_k})"
                    )
                except Exception as e:
                    logger.warning(
                        f"[serde] Type handler deserialization failed for {key} ({type_k}): {e}"
                    )
            else:
                logger.warning(
                    f"[serde] No type handler registered for {type_k} — cannot restore {key}"
                )

    if restored:
        executor.send_variables(restored)
        logger.debug(f"[serde] Restored {len(restored)} executor variables")
