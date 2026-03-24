---

# Python code style:

- Loguru for logging at all times
  - Please centralize logging to be formatted in 
    - [section] <message>
    - Use fstrings
- DRY Arch
- Always to abstract to 2 layers if needed? Inherit is the best and make it clear
- This project must be typed, no exceptions 
- This project is in uv. Never add to pyproject.toml to add dependencies, use uv add
--- 

# duro: Technical Spec

## Overview

A Python library that adds Temporal durable execution to existing smolagents setups. Developers wrap their existing agent in `DurableAgent` and get crash recovery, state persistence, and replay without re-invoking the LLM.

**Target user:** A team already running smolagents + LiteLLM that wants to bolt on Temporal durability without rewriting their agent logic.

---

## Problem

smolagents agents are stateless across process boundaries. If a multi-step agent crashes mid-execution, all progress is lost — the LLM must be re-invoked from scratch, wasting tokens and time. There is no existing library that bridges smolagents and Temporal.

Two state surfaces must survive across boundaries:

1. **Memory steps** — the message history the LLM sees (TaskStep, ActionStep, PlanningStep dataclasses)
2. **Executor variable namespace** — the Python variables accumulated by CodeAgent's `LocalPythonExecutor` (DataFrames, computed results, imported modules)

---

## Developer Experience

### Before (plain smolagents)

```python
agent = CodeAgent(tools=[...], model=model, additional_authorized_imports=["pandas", "numpy"])
result = agent.run("Analyze this dataset")
```

### After (with temporal durability)

```python
from duro import DurableAgent

def make_agent():
    return CodeAgent(
        tools=[...],
        model=LiteLLMModel(model_id="deepseek/deepseek-chat", api_key="..."),
        additional_authorized_imports=["pandas", "numpy", "requests"],
    )

durable = DurableAgent(agent_factory=make_agent)

# In application code — starts a Temporal workflow
result = await durable.run("Analyze this dataset")
```

### Worker setup

```python
from duro import AgentActivities, AgentWorkflow

activities = AgentActivities(agent_factory=make_agent)

worker = Worker(
    client,
    task_queue="smolagents",
    workflows=[AgentWorkflow],
    activities=[activities.run_steps],
)
```

That's it. The library handles serialization, memory reconstruction, executor state snapshots, and the workflow loop internally.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Temporal Workflow (deterministic)                │
│                                                  │
│   params: task, max_cycles, steps_per_cycle      │
│                                                  │
│   loop:                                          │
│     output = await run_steps(input)              │
│     if output.done: return output.final_answer   │
│     input = output  ◄── carry forward state      │
│     cycle += 1                                   │
│     if cycle >= max_cycles: return timeout        │
│                                                  │
└─────────────────────────────────────────────────┘
         │
         │ activity call (recorded in event history)
         ▼
┌─────────────────────────────────────────────────┐
│ Temporal Activity: run_steps (non-deterministic) │
│                                                  │
│   1. agent = agent_factory()                     │
│   2. restore memory steps from input             │
│   3. restore executor state from input           │
│   4. run N agent.step() calls (LLM + execution)  │
│   5. serialize updated memory steps              │
│   6. serialize executor state (dill)             │
│   7. return serialized state + final_answer      │
│                                                  │
│   On Temporal replay: this is NOT re-executed.   │
│   Temporal returns the recorded output.          │
└─────────────────────────────────────────────────┘
```

### Key property

On replay, Temporal returns recorded activity outputs. The LLM is never re-called. The `agent_factory` is only invoked during first execution of each activity — it constructs a fresh, short-lived agent that is discarded after the activity completes.

---

## Modules

### 1. `serde.py` — Serialization layer

Handles round-trip conversion of smolagents internals to/from JSON-safe dicts.

#### Memory step serialization

**Design principle:** Only serialize fields that `to_messages()` reads, because that's what the LLM sees. Everything else is optional metadata.

| Step type | Fields required for `to_messages()` | Optional metadata |
|-----------|--------------------------------------|-------------------|
| `TaskStep` | `task` | `task_images` (dropped — not JSON-safe, PIL objects) |
| `ActionStep` | `model_output`, `tool_calls`, `observations`, `error` | `code_action`, `action_output`, `is_final_answer`, `token_usage`, `step_number` |
| `PlanningStep` | `model_output_message`, `plan` | `model_input_messages` (rebuilt by agent), `token_usage` |

Fields that are explicitly **not serialized**:
- `model_input_messages` — rebuilt by `write_memory_to_messages()` before each LLM call
- `observations_images` / `task_images` — PIL Image objects, not JSON-serializable, set to `None` on restore
- `timing` — reconstructed as `Timing(start_time=0, end_time=0)` (monitoring only)

**Validated:** `to_messages()` output is identical between original and deserialized steps (tested with smolagents 1.24.0).

#### Executor state serialization (CodeAgent only)

**Why this is critical:** The `LocalPythonExecutor` maintains a single `self.state` dict for the entire agent run. Variables assigned in step 2 are live Python objects in step 6 — the LLM chains variable names across steps by design. This is not optional for non-trivial workloads. Example:

```python
# Step 2
series = get_nass_series_list(commodity="corn")   # → 40-row result, truncated in observation text
# Step 3 — can't reconstruct series from truncated text, must reuse the variable
series_id = [s for s in series if "yield" in s["description"].lower()][0]["id"]
data = get_nass_series_data(series_id=series_id)
# Step 5 (after replan)
result = analyze_multiple_timeseries(data)        # data is the live object, not reprinted text
```

For small scalars (a date, a float) the LLM sometimes hardcodes from observation text, but for anything structured (DataFrames, lists of dicts, API responses), it relies on the live variable.

Uses `dill` to serialize the `executor.state` dict — everything the agent's code has defined or imported.

```python
# What executor.state looks like after a pandas-heavy step:
{
    '__name__': '__main__',           # skipped (internal)
    '_print_outputs': PrintContainer,  # skipped (internal)
    '_operations_count': dict,         # skipped (internal)
    'pd': <module 'pandas'>,          # serialized (dill handles modules)
    'df': DataFrame,                   # serialized (pickle-native)
    'result': 42,                      # serialized
}
```

Internal keys (`__name__`, `_print_outputs`, `_operations_count`) are excluded. User-defined variables are serialized.

Restoration uses `executor.send_variables(restored_dict)` (smolagents ≥1.20).

Serialized format: `base64(dill.dumps(user_state))` — stored as a string in the Temporal activity output.

#### Type handler registry

Most objects from the approved import list serialize cleanly with dill. Two types fail because of lxml internals:

| Type | dill result | Reason |
|------|-------------|--------|
| `docx.document.Document` | FAIL | `CT_Document` contains lxml element tree, can't pickle |
| `pptx.presentation.Presentation` | FAIL | `CT_Presentation` contains lxml element tree, can't pickle |

**Everything else from the approved imports serializes fine:** all modules, pandas DataFrames/Series, numpy arrays/scalars, openpyxl Workbooks, matplotlib Figures, fitted sklearn models, scipy sparse matrices, datetime objects, compiled regex, user-defined functions/lambdas/classes, tempfile handles.

**Solution: save-to-bytes reconstruction.** Both `docx.Document` and `pptx.Presentation` support `save(BytesIO)` and can be reconstructed from the bytes. The round-trip preserves all content (paragraphs, tables, slides, shapes). Validated:

```python
# docx.Document: 36KB as bytes, full fidelity
buf = io.BytesIO(); doc.save(buf)
restored = docx.Document(io.BytesIO(buf.getvalue()))  # paragraphs, tables intact

# pptx.Presentation: 28KB as bytes, full fidelity
buf = io.BytesIO(); prs.save(buf)
restored = pptx.Presentation(io.BytesIO(buf.getvalue()))  # slides, shapes intact
```

The library ships with a type handler registry — a dict of type → `(serialize_fn, deserialize_fn)` pairs:

```python
# Built-in handlers
TYPE_HANDLERS = {
    "docx.document.Document": (
        lambda v: base64.b64encode(save_to_bytes(v)).decode(),
        lambda b: docx.Document(io.BytesIO(base64.b64decode(b))),
    ),
    "pptx.presentation.Presentation": (
        lambda v: base64.b64encode(save_to_bytes(v)).decode(),
        lambda b: pptx.Presentation(io.BytesIO(base64.b64decode(b))),
    ),
}
```

**Serialization flow for executor state:**

1. Split `executor.state` into internal keys (skip) and user keys
2. For each user variable, check the type handler registry — if a handler exists, use it
3. Wrap remaining variables in `dill.dumps()`
4. If dill fails on a specific variable: drop it, log a warning (agent self-corrects by recomputing on next step)
5. Encode as base64 for JSON transport

**Extensibility:** Developers can register custom handlers for their own types:

```python
from duro import register_type_handler

register_type_handler(
    MyCustomType,
    serialize_fn=lambda v: v.to_json(),
    deserialize_fn=lambda b: MyCustomType.from_json(b),
)
```

#### Serialization risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Undiscovered unpicklable type in executor state | Low — all approved imports tested, only docx/pptx fail and have handlers | Graceful degradation: drop variable, log warning. Agent self-corrects by recomputing on next step (costs 1 extra LLM call). |
| smolagents changes step dataclass fields between versions | Low-medium (API marked experimental) | Pin smolagents version. Round-trip test in CI catches breakage. |
| Large executor state (huge DataFrames) exceeding Temporal payload limits | Low (Temporal default max is 2MB for activity output) | Log payload size. If over threshold, fail loud in v0.1. Blob storage backend in v0.2. |
| `action_output` contains non-JSON-safe types | Common | Best-effort `str()` fallback. Not critical — `to_messages()` uses `observations` string, not `action_output`. |

#### Validated dill compatibility matrix (approved imports)

| Category | Types tested | Result |
|----------|-------------|--------|
| Modules | json, pandas, numpy, openpyxl, docx, pptx, os, datetime, tempfile, calendar, matplotlib, seaborn, scipy, sklearn, pdfplumber | All OK (dill handles modules natively) |
| pandas | DataFrame, Series, Index, Timestamp | All OK |
| numpy | ndarray (float64, int64), float64, int64, matrix | All OK |
| openpyxl | Workbook, Worksheet | All OK |
| docx | Document | FAIL → type handler (save-to-bytes) |
| pptx | Presentation | FAIL → type handler (save-to-bytes) |
| matplotlib | Figure, Axes | All OK |
| scipy | stats distributions, sparse matrices, fit results | All OK |
| sklearn | LinearRegression (fitted), StandardScaler (fitted) | All OK |
| stdlib | datetime, date, timedelta, compiled regex, sets, tuples, frozensets | All OK |
| user-defined | functions, lambdas, class instances | All OK |

---

### 2. `activities.py` — Temporal activities

Single activity class: `AgentActivities`.

#### Constructor

```python
AgentActivities(agent_factory: Callable[[], MultiStepAgent])
```

Takes a zero-argument callable that returns a fresh agent. Called once per activity execution. The factory pattern is necessary because agents are not serializable and must be constructed on the worker.

#### `run_steps` activity

**Input dataclass: `RunStepsInput`**

| Field | Type | Description |
|-------|------|-------------|
| `task` | `str` | The task/prompt |
| `steps` | `list[dict]` | Serialized memory steps from previous cycle (empty on first cycle) |
| `executor_state` | `str \| None` | Base64-encoded dill state (CodeAgent only) |
| `steps_to_run` | `int` | Number of `agent.step()` calls this cycle (default: 5) |
| `global_step_number` | `int` | Step counter carried across cycles (starts at 1). Used for planning interval checks. |

**Output dataclass: `RunStepsOutput`**

| Field | Type | Description |
|-------|------|-------------|
| `steps` | `list[dict]` | Updated serialized memory steps |
| `executor_state` | `str \| None` | Updated executor state |
| `final_answer` | `Any` | The agent's final answer, if produced |
| `done` | `bool` | Whether the agent has finished |
| `steps_completed` | `int` | How many steps actually ran this cycle |
| `error` | `str \| None` | Last error message, if any |
| `global_step_number` | `int` | Updated step counter for next cycle |

**Execution flow:**

1. Call `agent_factory()` to get a fresh agent
2. For CodeAgent: call `agent.python_executor.send_tools({**agent.tools, **agent.managed_agents})`
3. Restore memory: `agent.memory.steps = deserialize_steps(input.steps)`
4. Restore executor state: `deserialize_executor_state(agent.python_executor, input.executor_state)`
5. If no TaskStep exists in memory, append one
6. Loop `steps_to_run` times, tracking `global_step = input.global_step_number + i`:
   - **If `planning_interval` is set and this step is a planning step:** call `agent._generate_planning_step()`, append PlanningStep to memory
   - Create `ActionStep` with current global step number
   - Call `agent.step(memory_step)` — **this is the LLM call**
   - Append step to memory
   - If `result.is_final_answer`: break
   - On exception: log, append step anyway (agent may self-correct)
7. Serialize memory steps and executor state
8. Return output with updated `global_step_number`

**Error handling:** Exceptions during `agent.step()` are caught and stored in the step's error field. The agent's built-in self-correction mechanism handles these on the next step. The activity only raises to Temporal if something truly unrecoverable happens (e.g., agent_factory fails).

#### Agent type detection

The activity detects whether the agent is a `CodeAgent` or `ToolCallingAgent`:
- **CodeAgent:** Has `python_executor`. Executor state is serialized/restored.
- **ToolCallingAgent:** No executor. `executor_state` is always `None`.

Both types use the same memory step serialization.

---

### 3. `workflows.py` — Temporal workflow

#### `AgentWorkflow`

**Params dataclass: `AgentWorkflowParams`**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task` | `str` | required | The task/prompt |
| `max_cycles` | `int` | `10` | Max activity invocations before timeout |
| `steps_per_cycle` | `int` | `5` | Agent steps per activity call |
| `activity_timeout_seconds` | `int` | `300` | Timeout per activity (accounts for LLM latency) |

**Workflow logic (deterministic):**

```
initialize: steps=[], executor_state=None, cycle=0

while cycle < max_cycles:
    output = await activity.run_steps(RunStepsInput(
        task=task,
        steps=steps,
        executor_state=executor_state,
        steps_to_run=steps_per_cycle,
    ))

    steps = output.steps
    executor_state = output.executor_state

    if output.done:
        return AgentWorkflowResult(answer=output.final_answer, total_steps=count_steps(steps))

    cycle += 1

return AgentWorkflowResult(answer=None, timed_out=True, total_steps=count_steps(steps))
```

This is the only workflow. It's deliberately simple — the complexity lives in the activity.

---

### 4. `durable_agent.py` — Developer-facing wrapper

#### `DurableAgent`

The high-level wrapper that hides Temporal internals.

```python
class DurableAgent:
    def __init__(
        self,
        agent_factory: Callable[[], MultiStepAgent],
        task_queue: str = "smolagents",
        max_cycles: int = 10,
        steps_per_cycle: int = 5,
        activity_timeout_seconds: int = 300,
    ): ...

    async def run(
        self,
        task: str,
        temporal_client: Client | None = None,
        workflow_id: str | None = None,  # auto-generated if not provided
    ) -> Any:
        """Run the agent as a Temporal workflow. Returns the final answer."""
        ...

    def create_worker(self, client: Client) -> Worker:
        """Create a Temporal worker pre-configured for this agent."""
        ...
```

**`run()` behavior:**
1. Connects to Temporal (uses provided client or connects to localhost:7233)
2. Starts an `AgentWorkflow` with the given task
3. Waits for completion
4. Returns `final_answer`

**`create_worker()` behavior:**
1. Instantiates `AgentActivities` with the factory
2. Returns a `Worker` bound to the task queue with the workflow and activities registered

---

## Payload size considerations

Based on validated tests:

| Content | Typical size |
|---------|-------------|
| Simple math step (serialized) | ~570 bytes |
| Pandas-heavy step (serialized) | ~2.5 KB |
| Executor state with DataFrame | ~1.8 KB |
| 10-step conversation total | ~10-25 KB |

Temporal's default max payload is 2MB. This is well within limits for typical agent workloads. For agents that accumulate very large DataFrames in the executor, the library should log a warning when serialized state exceeds a configurable threshold (default: 1MB) and document the option of external blob storage.

---

## HTTP / approved imports concern

If the agent's approved import list includes `requests`, `httpx`, `urllib`, etc., the code agent can make HTTP calls. This creates two issues:

1. **Executor state may contain unpicklable HTTP objects** (response objects, session pools, connection objects). Mitigation: the graceful degradation in `serialize_executor_state` handles this — unpicklable values are dropped with a warning.

2. **Replay semantics for HTTP calls.** Temporal does NOT replay the code execution — it replays the *activity output*. So HTTP calls made inside `agent.step()` are part of the activity's first execution and their results are captured in the step's `observations` string. On replay, the recorded observations are returned without re-executing. This is correct behavior.

3. **Idempotency of tool calls.** If a tool call has side effects (e.g., POST to an API) and the activity is retried by Temporal (due to timeout/crash), the tool call will execute again. This is a standard Temporal concern, not specific to this library. Document it and recommend idempotency keys for side-effecting tools.

---

## Testing strategy

### Unit tests (no LLM, no Temporal)

- `test_serde_round_trip`: Serialize/deserialize each step type, assert `to_messages()` output matches
- `test_serde_with_error`: ActionStep with AgentError survives round-trip
- `test_executor_state_with_pandas`: dill round-trip with DataFrames, numpy arrays, modules
- `test_executor_state_graceful_degradation`: unpicklable values are dropped, picklable values survive
- `test_executor_state_empty`: None input/output handled correctly

### Integration tests (with LLM, no Temporal)

- `test_manual_step_restore`: Run 1 step, serialize, create new agent, restore, run step 2 — verify continuation
- `test_multi_step_handoff`: Run 2 steps, serialize at boundary, restore, continue to completion
- `test_code_agent_with_pandas`: Full round-trip with pandas workload
- `test_tool_calling_agent`: Same tests but with ToolCallingAgent

### Temporal integration tests (with Temporal test server)

- `test_workflow_completes`: Simple task → final answer via workflow
- `test_workflow_multi_cycle`: Task requiring >1 cycle of steps
- `test_workflow_timeout`: max_cycles exceeded → timed_out result
- `test_activity_retry`: Activity fails once, retries, succeeds

### CI validation

- Pin smolagents version
- Run `test_serde_round_trip` on every PR — this catches smolagents API breakage

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `smolagents` | `>=1.20.0` | Agent framework |
| `temporalio` | `>=1.20.0` | Workflow orchestration |
| `dill` | `>=0.3.8` | Executor state serialization (handles modules, pandas, numpy) |

Optional (blob storage backends):

| Package | Version | Purpose |
|---------|---------|---------|
| `google-cloud-storage` | `>=2.0` | GCS state store |
| `boto3` | `>=1.28` | S3 state store |

Dev dependencies: `pytest`, `pytest-asyncio`

---

## Resolved: Planning interval across cycle boundaries

**Problem:** When using `planning_interval`, smolagents inserts PlanningSteps at regular intervals (e.g., every 3 action steps). The planning check lives in `_run_stream()`:

```python
self.step_number = 1  # always resets!
while ...:
    if self.planning_interval is not None and (
        self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
    ):
        # generate planning step
```

Our library calls `agent.step()` directly, which delegates to `_step_stream()`. **`_step_stream()` contains no planning logic.** Planning is entirely in the outer `_run_stream()` loop, which we bypass.

This means: **if the developer sets `planning_interval` on their agent, planning will silently not happen** when run through the Temporal activity.

**Solution for v0.1:** The activity must handle planning manually. It tracks the global step number (persisted across cycles via `RunStepsInput`) and calls `agent._generate_planning_step()` at the correct intervals:

```python
# Inside the activity's step loop
global_step = next_step_number + i

if agent.planning_interval is not None and (
    global_step == 1 or (global_step - 1) % agent.planning_interval == 0
):
    for element in agent._generate_planning_step(
        task=input.task,
        is_first_step=(global_step == 1),
        step=global_step,
    ):
        if isinstance(element, PlanningStep):
            agent.memory.steps.append(element)
```

`_generate_planning_step` is a public method on `MultiStepAgent`. It yields a `PlanningStep` that gets appended to memory. The `is_first_step` flag controls whether it generates an initial plan or an updated plan.

The `RunStepsInput` and `RunStepsOutput` dataclasses need an additional field:

```python
@dataclass
class RunStepsInput:
    ...
    global_step_number: int = 1  # tracks step counter across cycles
```

This is also needed for `remaining_steps` in the update plan prompt, which tells the LLM how many steps it has left.

**Using `_run_stream()` was considered and rejected** because it always resets `self.step_number = 1` regardless of `reset=False`, which would cause an initial planning step to fire on every cycle boundary.

---

## Resolved: Managed agents (multi-agent)

**Decision:** Support in v0.1.

Managed agents are sub-agents that the parent agent can call as tools. When a managed agent runs, it gets its own memory and (for CodeAgent) its own executor. The parent agent's memory contains the managed agent's output in the `observations` field of the action step — so the parent's memory serialization already captures the result.

However, managed agents are reconstructed from scratch by the `agent_factory`. They don't need their own memory persisted across cycles because they run to completion within a single parent step. The parent step's `observations` captures their output.

**What the library needs to handle:**
- The `agent_factory` must return an agent with managed agents already attached
- `agent.python_executor.send_tools({**agent.tools, **agent.managed_agents})` — managed agents must be included in the executor's tool namespace (this is already done in the activity setup)

No additional serialization is needed for managed agents in v0.1.

---

## Blob storage for large executor states (v0.1)

Temporal's default max payload is 2MB. Tested payload sizes for realistic workloads:

| Workload | Dill size | Base64 | % of 2MB |
|----------|-----------|--------|----------|
| Small summary (3 rows) | 958b | 1.3KB | 0.1% |
| USDA-style (1700 rows, 5 cols) | 39KB | 52KB | 2.5% |
| Financial (5000 rows, 20 cols) | 841KB | 1.1MB | 53.5% |
| Large dataset (50k rows, 10 cols) | 4.0MB | 5.3MB | **254% ⚠️** |
| Realistic multi-df state | 41KB | 55KB | 2.6% |
| State + fitted sklearn model + data | 88KB | 117KB | 5.6% |

Any agent working with datasets over ~5000 dense numerical rows will exceed the limit. This is a production reality, not an edge case.

### Design

The library uses a `StateStore` abstraction. Executor state and memory steps that exceed an inline threshold are offloaded to external storage. Only a reference (a key/URI) is stored in the Temporal payload.

```python
class StateStore(Protocol):
    """Backend for persisting large executor state blobs."""

    async def put(self, key: str, data: bytes) -> None:
        """Store a blob."""
        ...

    async def get(self, key: str) -> bytes:
        """Retrieve a blob."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a blob (cleanup after workflow completion)."""
        ...
```

### Built-in implementations

**`InlineStore` (default):** No external storage. State is stored directly in the Temporal payload. Raises `PayloadTooLargeError` if serialized state exceeds `max_inline_bytes` (default: 1.5MB, leaving headroom for step data).

**`GCSStateStore`:** Stores blobs in Google Cloud Storage.

```python
from duro import DurableAgent, GCSStateStore

store = GCSStateStore(bucket="my-agent-state", prefix="workflows/")
durable = DurableAgent(agent_factory=make_agent, state_store=store)
```

**`S3StateStore`:** Same interface, backed by S3. (Optional dependency: `boto3`.)

**`R2StateStore`:** Cloudflare R2 — S3-compatible, zero egress fees, 10GB free tier. Ideal default for production since agent state blobs are write-once-read-once-delete and R2's free tier covers 1M write operations/month. Uses `boto3` under the hood pointed at the Cloudflare S3-compatible endpoint.

```python
from duro import DurableAgent, R2StateStore

store = R2StateStore(
    account_id="your-cf-account-id",
    bucket="agent-state",
    access_key_id="...",
    secret_access_key="...",
    prefix="workflows/",
)
durable = DurableAgent(agent_factory=make_agent, state_store=store)
```

Internally, `R2StateStore` is a thin wrapper around `S3StateStore` with the endpoint set to `https://{account_id}.r2.cloudflarestorage.com`.

**`FileStateStore`:** Local filesystem, useful for development/testing.

```python
from duro import DurableAgent, FileStateStore

store = FileStateStore(directory="/tmp/agent-state")
durable = DurableAgent(agent_factory=make_agent, state_store=store)
```

### How it works in the activity

```python
# Serialize executor state
serialized = dill.dumps(user_state)

if len(serialized) <= max_inline_bytes:
    # Store directly in Temporal payload
    return RunStepsOutput(executor_state=base64.b64encode(serialized).decode(), ...)
else:
    # Offload to blob store
    key = f"{workflow_id}/{run_id}/executor_state/{cycle}"
    await state_store.put(key, serialized)
    return RunStepsOutput(executor_state_ref=key, ...)

# On restore
if output.executor_state:
    # Inline
    data = base64.b64decode(output.executor_state)
elif output.executor_state_ref:
    # Fetch from blob store
    data = await state_store.get(output.executor_state_ref)
```

### Updated dataclasses

```python
@dataclass
class RunStepsOutput:
    steps: list[dict[str, Any]]
    executor_state: str | None = None       # inline base64 (small payloads)
    executor_state_ref: str | None = None   # blob store key (large payloads)
    final_answer: Any = None
    done: bool = False
    steps_completed: int = 0
    error: str | None = None
    global_step_number: int = 1
```

### Cleanup

When the workflow completes (success or timeout), it should clean up blob store keys for that workflow. The workflow can emit a final cleanup step, or the `DurableAgent` wrapper can handle it in `run()` after the workflow returns.

### Dependencies

| Store | Additional dependency |
|-------|----------------------|
| `InlineStore` | None |
| `GCSStateStore` | `google-cloud-storage` |
| `S3StateStore` | `boto3` |
| `R2StateStore` | `boto3` (S3-compatible) |
| `FileStateStore` | None |

These are optional dependencies:

```toml
[project.optional-dependencies]
gcs = ["google-cloud-storage>=2.0"]
s3 = ["boto3>=1.28"]
r2 = ["boto3>=1.28"]  # same dep, separate extra for clarity
```

---

## Open questions (remaining)

1. **Stream outputs.** The current spec doesn't support streaming. `agent.step()` returns synchronously. If streaming is needed, the activity would need to use Temporal heartbeats to stream intermediate results. Punt to v0.2.

2. **Executor type.** The spec assumes `LocalPythonExecutor` (confirmed as the team's setup). If the agent uses Docker/E2B/Modal executors, the dill-based state serialization won't work (those are remote sandboxes with their own state). The library should detect non-local executors and raise a clear error in v0.1.
