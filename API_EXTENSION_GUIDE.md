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
The next question is generated after each answer; the backend does not pre-create the full question list.

---

## 1) Start Experiment

### Endpoint
`POST /research/start`

### Request Body

```json
{
  "prompt": "Build a hybrid quantum-classical classifier",
  "priority": "normal",
  "tags": ["quantum", "classification"],
  "config_overrides": {
    "random_seed": 42,
    "hardware_target": "cpu",
    "max_epochs": 20
  }
}
```

### Response `data` (important fields)

```json
{
  "experiment_id": "exp_20260224_ab12cd",
  "status": "waiting_user",
  "phase": "clarifier",
  "execution_target": "local_machine",
  "llm": {"provider": "rule_based", "model": "deterministic-orchestrator"},
  "pending_questions": {
    "mode": "sequential_dynamic",
    "current_question": {"id": "Q1", "text": "...", "type": "choice", "options": [".py", ".ipynb"]},
    "questions": [{"id": "Q1", "text": "...", "type": "choice", "options": [".py", ".ipynb"]}],
    "asked_question_ids": [],
    "answered_count": 0,
    "total_questions_planned": 0
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
  "alternative_preference": ""
}
```

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
  "llm_provider": "rule_based",
  "llm_model": "deterministic-orchestrator",
  "progress_pct": 85
}
```

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

## Extension Tips

- Keep new responses inside the existing envelope format.
- For new write operations, include explicit validation errors with stable error codes.
- For new long-running flows, keep status polling fields consistent (`status`, `phase`, `progress_pct`, `waiting_for_user`).
- If you add new agent actions/commands, log them in `experiment_logs.details` for traceability.
