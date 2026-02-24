# AI + Quantum Research Platform

Production-style backend for autonomous research orchestration (AI + Quantum), with:
- FastAPI API layer
- multi-phase agent workflow
- persistent ResearchState in SQLite
- structured logging
- strict validation and safety guardrails
- optional Hugging Face Qwen 7B integration
- asynchronous background workflow execution for heavy phases

## 1. Project features

- End-to-end research lifecycle:
  - clarification
  - planning
  - environment preparation
  - dataset generation/acquisition
  - code generation (+ optional quantum gate)
  - execution
  - error recovery
  - evaluation
  - report generation
- Security controls:
  - project-path whitelist checks
  - action schema validation
  - subprocess argument sanitization
  - retry caps
- Logging for all major layers:
  - HTTP request/response logs
  - phase start/end logs
  - DB operation logs
  - subprocess run logs
- RL feedback signals:
  - phase validation rewards
  - runtime success/failure rewards
  - user decision reward signals
  - latency, evaluation-quality, and terminal-outcome rewards/penalties
- Observability and persistence:
  - numeric metrics normalized into `experiment_metrics`
  - LLM usage telemetry (tokens, latency, estimated cost)
  - system-level phase latency histograms and failure clustering
- Hugging Face model support for master LLM provider (`Qwen 7B` family).

## 2. Services and main modules

- API: `src/api/*`
- Workflow runner: `src/graph/runner.py`
- Agents: `src/agents/*`
- Safety/validation/runtime: `src/core/*`
- Persistence: `src/db/*`
- LLM integrations: `src/llm/*`
- Config: `src/config/settings.py`

## 3. Setup guide

### 3.1 Prerequisites

- Python 3.11
- pip or Poetry

### 3.2 Install dependencies

Option A: `requirements.txt`
```bash
pip install -r requirements.txt
```

Option B: Poetry
```bash
poetry install
```

### 3.3 Environment config

Copy:
```bash
cp .env.example .env
```

Minimum `.env` for Hugging Face Qwen:
```env
MASTER_LLM_PROVIDER=huggingface
HF_API_KEY=hf_xxx
HF_INFERENCE_URL=https://router.huggingface.co/v1
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct

PROJECT_ROOT=./workspace/projects
STATE_DB_PATH=./workspace/state.db
LOG_LEVEL=INFO
```

If you want offline deterministic mode:
```env
MASTER_LLM_PROVIDER=rule_based
```

Optional adaptive/resilience toggles:
```env
EXPERIMENT_VENV_ENABLED=true
AUTO_CONFIRM_LOW_RISK=true
LOW_RISK_PACKAGES=numpy,pandas,matplotlib,scikit-learn,requests
WORKFLOW_BACKGROUND_ENABLED=true
METRICS_TABLE_ENABLED=true
FAILURE_INJECTION_ENABLED=false
FAILURE_INJECTION_RATE=0.0
FAILURE_INJECTION_POINTS=
AUTO_RETRY_ON_LOW_METRIC=true
MIN_PRIMARY_METRIC_FOR_SUCCESS=0.75
```

### 3.4 Run server

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Swagger:
- `http://localhost:8000/api/docs`

## 4. API usage and endpoint examples

Detailed extension guide (all endpoints, request/response formats):
- `API_EXTENSION_GUIDE.md`

Authentication:
- No API key is required for local endpoint access in this build.

Base URL: `http://localhost:8000/api/v1`  
Header on all requests:
```http
Content-Type: application/json
```

### 4.1 Start experiment

`POST /research/start`

Request:
```json
{
  "prompt": "Build a hybrid quantum-classical classifier for synthetic data",
  "priority": "normal",
  "tags": ["quantum", "classification"],
  "config_overrides": {
    "random_seed": 42,
    "hardware_target": "cpu",
    "max_epochs": 20
  }
}
```

Response (example):
```json
{
  "success": true,
  "data": {
    "experiment_id": "exp_20260224_abc123",
    "status": "waiting_user",
    "phase": "clarifier",
    "pending_questions": {
      "mode": "sequential_dynamic",
      "current_question": {"id":"Q1","type":"choice","text":"Do you want .py scripts or .ipynb notebooks?"},
      "questions": [
        {"id":"Q1","type":"choice","text":"Do you want .py scripts or .ipynb notebooks?"}
      ],
      "asked_question_ids": [],
      "answered_count": 0
    }
  }
}
```

### 4.2 Answer clarification questions

`POST /research/{experiment_id}/answer`

Sequential mode: submit one answer per request.

Request:
```json
{
  "answers": {
    "Q1": ".py"
  }
}
```

### 4.3 Confirm pending action

`POST /research/{experiment_id}/confirm`

Request:
```json
{
  "action_id": "act_1234abcd",
  "decision": "confirm",
  "reason": "Approve installation for test run",
  "alternative_preference": ""
}
```

### 4.4 Status / logs / results / report

- `GET /research/{experiment_id}/status`
- `GET /research/{experiment_id}/logs`
- `GET /research/{experiment_id}/results`
- `GET /research/{experiment_id}/report?format=markdown&download=false`

### 4.5 Files and system endpoints

- `GET /research/{experiment_id}/files`
- `GET /research/{experiment_id}/files/{file_path}`
- `GET /system/health`
- `GET /system/metrics`

## 5. End-to-end testing script (`test.py`)

Added root script `test.py` that tests the full flow and stores all outputs as JSON (including metrics).

### 5.1 Run

```bash
python test.py
```

Optional:
```bash
python test.py --scenarios standard,quantum,retry,abort --timeout-sec 600
python test.py --allow-remote-base-url --base-url http://127.0.0.1:8000/api/v1
```

### 5.2 What it does

- Runs multi-scenario flows (`standard`, `quantum`, `retry`, `abort`)
- Calls all API endpoints at least once across selected scenarios
- Stores each request and response (method/path/body/status/headers/latency) as JSON
- Auto-answers clarification questions and auto-confirms pending actions
- Stores detailed agent trace JSON per scenario:
  - clarification questions asked
  - answers sent to agents
  - confirmation actions
  - subprocess commands executed
  - API step-by-step responses
- Captures scenario-level metrics:
  - endpoint coverage
  - latency stats
  - agent-phase validation from logs
  - LLM provider/model used
  - RL/system metrics snapshot
- Enforces local execution by default (`localhost` only unless explicitly overridden)

### 5.3 Output location

By default:
- `workspace/test_outputs/<timestamp>/`

Contains:
- `00_manifest.json`
- `01_endpoint_coverage.json`
- one folder per scenario with:
  - `requests/*.json` (per-request artifacts with request body + response)
  - `90_flow.json` (scenario flow summary)
  - `95_agent_trace.json` (agent Q/A + command trace)
  - `99_summary.json` (scenario metrics and validation)
- `99_summary.json` (global summary across scenarios)

## 6. Logging

Structured logging is enabled via `structlog` and includes:
- `app.startup` / `app.shutdown`
- `http.request.start` / `http.request.end` / `http.request.error`
- `phase.start` / `phase.end`
- validation warnings/errors
- DB create/update/log operations
- subprocess start/end/error
- agent start/end logs

Set log level in `.env`:
```env
LOG_LEVEL=INFO
```

## 7. Notes

- If HF model endpoint is unavailable, the current implementation falls back to rule-based safe behavior.
- Keep secrets in `.env` only (never commit keys).
- Existing unit tests can still run:
```bash
pytest -q
```


## 8. Post-run analysis (metrics, endpoints, agent behavior, scenarios, latency)

Use `analyze_run.py` to generate a consolidated analysis file from:
- `workspace/state.db` (experiments, logs, RL feedback, metrics snapshots)
- `workspace/test_outputs/*` (endpoint trace JSON created by `test.py`)

### 8.1 Run

```bash
python analyze_run.py
```

Optional experiment filter:

```bash
python analyze_run.py --experiment-id exp_20260224_b1772d
```

Optional custom paths:

```bash
python analyze_run.py \
  --db-path workspace/state.db \
  --test-outputs-root workspace/test_outputs \
  --output-json workspace/analysis/analysis_summary.json \
  --output-md workspace/analysis/analysis_report.md
```

### 8.2 Output files

- JSON summary: `workspace/analysis/analysis_summary.json`
- Markdown report: `workspace/analysis/analysis_report.md`

### 8.3 What is included

- API endpoint coverage and status/error counts
- API latency stats (`X-Process-Time`) with avg/p95/min/max
- Agent/phase behavior from DB logs and phase completion events
- Scenario-level request patterns (clarification, confirmation, abort flows)
- Metrics aggregation from `state_json.metrics` and `experiment_metrics` table
- RL feedback aggregation (avg reward and positive rate by phase/signal)
- Coverage notes when data sources are missing
