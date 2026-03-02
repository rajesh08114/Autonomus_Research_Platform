# API Extension Guide

This guide is for users extending the research platform API.

Authentication:
- No API key is required for local endpoint access in this build.
- Heavy workflow phases run in background mode by default. Use status polling endpoints to track progress.

Base URL: `http://127.0.0.1:8000/api/v1`  
Headers on all requests:

```http
Content-Type: application/json
```

## Response Envelope

All API responses use:

```json
{
  "success": true,
  "data": {},
  "error": null,
  "request_id": "req_xxx",
  "timestamp": "2026-02-24T00:00:00+00:00",
  "version": "2.0.0"
}
```

When `success=false`, inspect `error.code`, `error.message`, `error.details`.

## Clarification Model (Sequential)

Clarifier is now one-question-at-a-time.

- `pending_questions.current_question`: the single active question
- `pending_questions.questions`: single-item list (same active question)
- `pending_questions.asked_question_ids`: IDs already answered
- `pending_questions.answered`: answered history

`POST /research/{experiment_id}/answer` accepts exactly one answer each call.
The backend keeps a dynamic question plan generated from the prompt and advances one question at a time.

---

## 1) Start Experiment

### Endpoint
`POST /research/start`

Domain guard:
- Domain is classified by the backend LLM (`ai`, `quantum`, or `unsupported`).
- Only prompts classified as AI/ML or Quantum are accepted.
- Non-supported domains return `400` with code `UNSUPPORTED_RESEARCH_DOMAIN`.
- If classifier service is temporarily unavailable, start returns `503` with code `DOMAIN_CLASSIFIER_UNAVAILABLE`.

### Request Body

```json
{
  "prompt": "Build a hybrid quantum-classical classifier",
  "research_type": "quantum",
  "priority": "normal",
  "tags": ["quantum", "classification"],
  "user_id": "alice",
  "test_mode": false,
  "config_overrides": {
    "random_seed": 42,
    "hardware_target": "cpu",
    "max_epochs": 20,
    "default_allow_research": true
  }
}
```

`config_overrides.default_allow_research=true` enables fully automatic flow:
- Clarification questions are auto-answered with safe defaults.
- Pending approval actions are auto-confirmed in non-local execution modes.
- In VS Code local mode, extension auto-runs pending local actions and posts confirmations.

### Response `data` (important fields)

```json
{
  "experiment_id": "exp_20260224_ab12cd",
  "status": "waiting_user",
  "phase": "clarifier",
  "research_type": "quantum",
  "research_scope": {"user_id": "alice", "test_mode": false, "collection_key": "user:alice"},
  "execution_target": "local_machine",
  "execution_mode": "vscode_extension",
  "default_allow_research": true,
  "llm": {"provider": "huggingface", "model": "Qwen/Qwen2.5-7B-Instruct"},
  "pending_questions": {
    "mode": "sequential_dynamic",
    "current_question": {"id": "Q1", "topic": "output_format", "text": "...", "type": "choice", "options": [".py", ".ipynb", "hybrid"]},
    "questions": [{"id": "Q1", "topic": "output_format", "text": "...", "type": "choice", "options": [".py", ".ipynb", "hybrid"]}],
    "asked_question_ids": [],
    "answered_count": 0,
    "total_questions_planned": 7
  }
}
```

---

## 2) Answer Clarification Question (Single Answer)

### Endpoint
`POST /research/{experiment_id}/answer`

### Request Body (one key only)

```json
{
  "answers": {
    "Q1": ".py"
  }
}
```

### Response `data` (if more questions remain)

```json
{
  "experiment_id": "exp_20260224_ab12cd",
  "research_type": "quantum",
  "answers_received": 1,
  "answered_question_ids": ["Q1"],
  "status": "waiting_user",
  "phase": "clarifier",
  "pending_questions": {
    "current_question": {"id": "Q2", "text": "...", "type": "choice"},
    "questions": [{"id": "Q2", "text": "...", "type": "choice"}]
  },
  "question_progress": {"answered_count": 1, "total_planned": 8},
  "next_action": "answer_next_question"
}
```

### Response `data` (when clarification is complete)

```json
{
  "status": "running",
  "phase": "planner",
  "pending_questions": null,
  "next_action": "wait"
}
```

---

## 3) Confirm Pending Action

### Endpoint
`POST /research/{experiment_id}/confirm`

### Request Body

```json
{
  "action_id": "act_1234abcd",
  "decision": "confirm",
  "reason": "Approve install",
  "alternative_preference": "",
  "execution_result": {
    "returncode": 0,
    "stdout": "...",
    "stderr": "",
    "duration_sec": 1.1,
    "command": ["python", "-m", "pip", "install", "numpy==1.26.4"],
    "cwd": "C:/.../exp_...",
    "created_files": []
  }
}
```

Note: for local actions (`apply_file_operations`, `prepare_venv`, `install_package`, `run_local_commands`), send `execution_result` when `decision="confirm"`.

### Response `data`

```json
{
  "experiment_id": "exp_20260224_ab12cd",
  "action_id": "act_1234abcd",
  "decision": "confirm",
  "status": "running",
  "phase": "env_manager",
  "pending_action": null
}
```

---

## 3b) Update Selected Fields On Existing Experiment

Use this when a user wants to change only specific fields on an already-created experiment.

### Endpoint
`PATCH /research/{experiment_id}`

### Request Body

```json
{
  "updates": {
    "target_metric": "f1_macro",
    "framework": "xgboost",
    "research_plan": {"problem_type": "classification"}
  },
  "merge_nested": true
}
```

Notes:
- `updates` is an allowlisted partial update payload.
- Nested dict fields (`research_plan`, `clarifications`) are merged when `merge_nested=true`.
- Running experiments reject updates with `409` to avoid state races.

### Response `data` (key fields)

```json
{
  "experiment_id": "exp_20260224_ab12cd",
  "status": "waiting_user",
  "phase": "clarifier",
  "updated_fields": ["framework", "research_plan", "target_metric"],
  "applied_updates": {"framework": "xgboost", "target_metric": "f1_macro"},
  "rejected_fields": {}
}
```

---

## 4) Experiment Details

### Endpoint
`GET /research/{experiment_id}`

### Response `data` (key fields)

```json
{
  "experiment_id": "exp_20260224_ab12cd",
  "status": "success",
  "phase": "finished",
  "execution_target": "local_machine",
  "execution_mode": "vscode_extension",
  "llm": {"provider": "huggingface", "model": "Qwen/Qwen2.5-7B-Instruct"},
  "created_files": [],
  "metrics": {}
}
```

---

## 5) Status

### Endpoint
`GET /research/{experiment_id}/status`

### Response `data` (key fields)

```json
{
  "experiment_id": "exp_20260224_ab12cd",
  "status": "running",
  "phase": "subprocess_runner",
  "waiting_for_user": false,
  "pending_action": null,
  "execution_target": "local_machine",
  "execution_mode": "vscode_extension",
  "llm_provider": "huggingface",
  "llm_model": "Qwen/Qwen2.5-7B-Instruct",
  "progress_pct": 85
}
```

When waiting for local execution, `pending_action` includes:
- `commands`: command array to run on user device
- `cwd`: working directory
- `file_operations`: files the extension should create/update before running commands
- `timeout_seconds`
- `action_id` used in `/confirm`
- `action` can be:
  - `apply_file_operations` (dataset/codegen/quantum phases)
  - `prepare_venv` / `install_package` (env phase)
  - `run_local_commands` (execution phase)

---

## 6) Logs

### Endpoint
`GET /research/{experiment_id}/logs?limit=100&offset=0`

### Response `data` (key fields)

```json
{
  "logs": [
    {
      "phase": "clarifier",
      "level": "info",
      "message": "Clarification answer received; next question queued",
      "details": {"answer": {"id": "Q1", "value": ".py"}, "next_question": {"id": "Q2"}}
    },
    {
      "phase": "subprocess_runner",
      "message": "Subprocess execution started",
      "details": {"command": ["C:/.../python.exe", "C:/.../main.py"], "cwd": "C:/..."}
    }
  ],
  "execution_logs": [
    {
      "script_path": "C:/.../main.py",
      "command": ["C:/.../python.exe", "C:/.../main.py"],
      "cwd": "C:/.../exp_xxx",
      "returncode": 0,
      "duration_sec": 0.2
    }
  ]
}
```

---

## 7) Results

### Endpoint
`GET /research/{experiment_id}/results`

### Response `data`

Model/evaluation metrics payload, for example:

```json
{
  "evaluation": {"accuracy": 0.91, "f1_macro": 0.90},
  "training": {"duration_sec": 2.4, "epochs": 20}
}
```

---

## 8) Report

### Endpoint
`GET /research/{experiment_id}/report?format=markdown&download=false`

### Response `data` (key fields)

```json
{
  "report_path": "C:/.../docs/final_report.md",
  "word_count": 1200,
  "sections": ["abstract", "objective"],
  "content": "# Abstract ..."
}
```

---

## 9) Files

### List Files
`GET /research/{experiment_id}/files`

### Get File Content
`GET /research/{experiment_id}/files/{file_path}`

### Get Plot
`GET /research/{experiment_id}/plots/{plot_name}`

---

## 10) List / Abort / Retry

### List
`GET /research?limit=20&offset=0`

### Abort
`DELETE /research/{experiment_id}/abort`

Request:

```json
{
  "reason": "User requested cancellation",
  "save_partial": true
}
```

### Retry
`POST /research/{experiment_id}/retry`

Request:

```json
{
  "from_phase": "error_recovery",
  "reset_retries": true,
  "override_config": {"hardware_target": "cpu"}
}
```

---

## 11) System Endpoints

### Health
`GET /system/health`

### Metrics
`GET /system/metrics`

Includes:
- experiment counts and success rate
- latency/performance aggregates
- llm usage distribution
- RL feedback aggregates
- execution target (`local_machine`)

---

## 12) Chat Endpoints (History-Aware Research Copilot)

Chat endpoint answers user questions using scoped research history:
- `test_mode=true`: reads from unified test collection (`test:unified`)
- `test_mode=false`: reads user-scoped collection (`user:<user_id>`)

### Ask Chat Assistant
`POST /chat/research`

Request:
```json
{
  "message": "How should I improve preprocessing and plots?",
  "user_id": "alice",
  "test_mode": false,
  "context_limit": 5
}
```

Response `data` (key fields):
```json
{
  "answer": "....",
  "follow_up_questions": ["..."],
  "scope": {"user_id": "alice", "test_mode": false, "collection_key": "user:alice"},
  "retrieval": {
    "history_source": "collection",
    "history_loaded": 3,
    "history_used": 3,
    "references": [{"experiment_id": "exp_..."}]
  },
  "generation": {"provider": "huggingface", "strategy": "history_grounded_chat_completion"},
  "token_usage": {"prompt_tokens": 21, "completion_tokens": 75, "total_tokens": 96}
}
```

### Read Chat History
`GET /chat/history?user_id=alice&test_mode=false&limit=40`

---

## Extension Tips

- Keep new responses inside the existing envelope format.
- For new write operations, include explicit validation errors with stable error codes.
- For new long-running flows, keep status polling fields consistent (`status`, `phase`, `progress_pct`, `waiting_for_user`).
- If you add new agent actions/commands, log them in `experiment_logs.details` for traceability.
