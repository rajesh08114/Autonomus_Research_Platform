# AI + Quantum Research Platform

LLM-orchestrated research backend (FastAPI + multi-phase agents) with a VS Code local execution contract.

## 1. What It Does

The platform runs an experiment through these phases:

1. `clarifier`
2. `planner`
3. `env_manager`
4. `dataset_manager`
5. `code_generator`
6. `quantum_gate` (only when required)
7. `job_scheduler`
8. `subprocess_runner`
9. `results_evaluator`
10. `doc_generator`

Core behavior:

- Dynamic LLM planning/generation across phases.
- Strict state-only validation for code generation.
- Deterministic/safe local execution for env setup, file writes, and command runs.
- Structured metrics, logs, and final report generation.

## 2. Repository Layout

- API: `src/api/*`
- Agent phases: `src/agents/*`
- Workflow runner: `src/graph/runner.py`
- LLM adapters/parsing: `src/llm/*`
- Core runtime/safety: `src/core/*`
- Persistence: `src/db/*`
- Config: `src/config/settings.py`

## 3. Setup

### 3.1 Prerequisites

- Python 3.10+ (3.11 recommended)
- `pip`

### 3.2 Create Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Configure Environment

```bash
cp .env.example .env
```

Required keys in `.env`:

```env
MASTER_LLM_PROVIDER=huggingface
HF_API_KEY=hf_xxx
HF_INFERENCE_URL=https://router.huggingface.co/v1
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
```

Important notes:

- Keep secrets only in `.env`.
- `HF_MODEL_ID` must be accessible by your HF token; otherwise startup will fail.

### 3.5 Start Server

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Docs:

- Swagger: `http://127.0.0.1:8000/api/docs`

## 4. VS Code Extension Integration Contract

Set:

```env
EXECUTION_MODE=vscode_extension
LOCAL_PYTHON_COMMAND=python
```

When running in VS Code mode, backend phases emit `pending_action` payloads.  
Your extension must execute those actions locally and then confirm back to API.

Supported action types:

- `prepare_venv`
- `install_package`
- `apply_file_operations`
- `run_local_commands`

### 4.1 Required Extension Loop

1. Poll `GET /api/v1/research/{experiment_id}/status`.
2. If `status=waiting_user` and `pending_action` exists:
   - Apply all `pending_action.file_operations` locally.
   - Execute `pending_action.command` or `pending_action.commands` in `pending_action.cwd`.
   - Capture `returncode`, `stdout`, `stderr`, `duration_sec`, `command`, `cwd`, `created_files`.
   - Call `POST /api/v1/research/{experiment_id}/confirm`.
3. Repeat until phase reaches `finished` / experiment reaches terminal status.

### 4.2 Confirm Payload Example

```json
{
  "action_id": "act_ab12cd34",
  "decision": "confirm",
  "reason": "Executed locally",
  "alternative_preference": "",
  "execution_result": {
    "returncode": 0,
    "stdout": "ok",
    "stderr": "",
    "duration_sec": 1.42,
    "command": ["python", "main.py"],
    "cwd": "C:/path/to/project",
    "created_files": ["C:/path/to/project/outputs/metrics.json"]
  }
}
```

## 5. API Flow (End-to-End)

### 5.1 Start Experiment

`POST /api/v1/research/start`

```json
{
  "prompt": "Build a classifier on synthetic data",
  "research_type": "ai",
  "config_overrides": {
    "execution_mode": "vscode_extension",
    "random_seed": 42,
    "max_epochs": 20
  }
}
```

### 5.2 Clarification

Respond with `POST /api/v1/research/{experiment_id}/answer` until clarifier is complete:

```json
{
  "answers": {
    "Q1": ".py"
  }
}
```

### 5.3 Execute Pending Local Actions

Use the VS Code loop in section 4 until completion.

### 5.4 Collect Outputs

- Status: `GET /api/v1/research/{experiment_id}/status`
- Logs: `GET /api/v1/research/{experiment_id}/logs`
- Metrics: `GET /api/v1/research/{experiment_id}/results`
- Report: `GET /api/v1/research/{experiment_id}/report?format=markdown`
- Files: `GET /api/v1/research/{experiment_id}/files`

## 6. Local Paths and Artifacts

Per experiment (under `PROJECT_ROOT`):

- Dataset: `data/raw/dataset.csv`
- Dataset validation: `data/validate_data.py`
- Data report: `data/data_report.json`
- Generated code entrypoint: `main.py`
- Runtime metrics: `outputs/metrics.json`
- Final documentation: `docs/final_report.md`

## 7. Running the Built-in Test Harness

Run:

```bash
python test.py
```

Runs multi-scenario API flow and stores trace output under:

- `workspace/test_outputs/*`

Then analyze:

```bash
python analyze_run.py
```

## 8. Troubleshooting

### Startup fails with `403 Forbidden` from HF

Check:

1. `HF_API_KEY` is valid and active.
2. `HF_MODEL_ID` is available to that token.
3. Base URL is exactly `https://router.huggingface.co/v1`.
4. No stale token values in shell/session override `.env`.

### Backend waits for user confirmation forever

Your extension likely did not:

1. materialize `file_operations`, or
2. execute command(s), or
3. post `confirm` with valid `execution_result`.

## 9. Development Validation

Run tests:

```bash
pytest -q
```

