from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import requests

EXPERIMENT_ID_RE = re.compile(r"exp_\d{8}_[0-9a-f]+", re.IGNORECASE)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
EXPECTED_AGENTS = [
    "clarifier",
    "planner",
    "env_manager",
    "dataset_manager",
    "code_generator",
    "quantum_gate",
    "job_scheduler",
    "subprocess_runner",
    "error_recovery",
    "results_evaluator",
    "doc_generator",
]


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def _safe_json(response: requests.Response) -> Any:
    content_type = str(response.headers.get("content-type", "")).lower()
    if "application/json" in content_type:
        try:
            return response.json()
        except Exception:
            return {"raw_text": response.text[:10000]}
    if "text/" in content_type:
        return {"raw_text": response.text[:10000]}
    return {"raw_bytes_len": len(response.content), "content_type": content_type}


def _extract_data(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            return data
    return {}


def _normalize_endpoint(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if "/api/v1/" in path:
        path = path.split("/api/v1", 1)[1]
    path = re.sub(
        r"/research/exp_\d{8}_[0-9a-f]+/files/.+",
        "/research/{experiment_id}/files/{file_path}",
        path,
        flags=re.IGNORECASE,
    )
    path = re.sub(
        r"/research/exp_\d{8}_[0-9a-f]+/plots/.+",
        "/research/{experiment_id}/plots/{plot_name}",
        path,
        flags=re.IGNORECASE,
    )
    path = EXPERIMENT_ID_RE.sub("{experiment_id}", path)
    if parsed.query:
        keys = sorted({k for k, _ in parse_qsl(parsed.query, keep_blank_values=True)})
        if keys:
            return f"{path}?{'&'.join(keys)}"
    return path


def _latency_stats(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"count": 0.0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))
    return {
        "count": float(len(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "avg_ms": float(sum(ordered) / len(ordered)),
        "median_ms": float(statistics.median(ordered)),
        "p95_ms": float(ordered[p95_index]),
    }


def _default_answer(question: dict[str, Any]) -> Any:
    qid = str(question.get("id", ""))
    topic = str(question.get("topic", "")).strip().lower()
    qtype = str(question.get("type", ""))
    options = question.get("options") or []
    default = question.get("default")
    if default is not None:
        return default
    topic_fallback = {
        "output_format": ".py",
        "algorithm_class": "supervised",
        "requires_quantum": False,
        "quantum_framework": "pennylane",
        "quantum_algorithm": "VQE",
        "quantum_qubit_count": 4,
        "quantum_backend": "default.qubit",
        "dataset_source": "synthetic",
        "kaggle_dataset_id": "owner/dataset",
        "target_metric": "accuracy",
        "problem_type": "classification",
        "code_level": "intermediate",
        "hardware_target": "cpu",
        "framework_preference": "auto",
        "python_version": "3.11",
        "random_seed": 42,
        "max_epochs": 20,
        "auto_retry_preference": "enabled",
    }
    if topic in topic_fallback:
        return topic_fallback[topic]
    fallback = {
        "Q_OUTPUT_FORMAT": ".py",
        "Q_ALGORITHM_CLASS": "supervised",
        "Q_REQUIRES_QUANTUM": False,
        "Q_QUANTUM_FRAMEWORK": "pennylane",
        "Q_QUANTUM_ALGORITHM": "VQE",
        "Q_QUANTUM_QUBIT_COUNT": 4,
        "Q_QUANTUM_BACKEND": "default.qubit",
        "Q_DATASET_SOURCE": "synthetic",
        "Q_KAGGLE_DATASET_ID": "owner/dataset",
        "Q_TARGET_METRIC": "accuracy",
        "Q_PROBLEM_TYPE": "classification",
        "Q_CODE_LEVEL": "intermediate",
        "Q_HARDWARE_TARGET": "cpu",
        "Q_FRAMEWORK_PREFERENCE": "auto",
        "Q_PYTHON_VERSION": "3.11",
        "Q_RANDOM_SEED": 42,
        "Q_MAX_EPOCHS": 20,
        "Q_AUTO_RETRY_PREFERENCE": "enabled",
    }
    if qid in fallback:
        return fallback[qid]
    if qtype == "boolean":
        return False
    if qtype == "number":
        return 1
    if qtype == "choice" and options:
        return options[0]
    return "auto"


def _enforce_local_base_url(base_url: str, allow_remote: bool) -> None:
    host = str(urlparse(base_url).hostname or "").lower()
    if not allow_remote and host not in LOCAL_HOSTS:
        raise RuntimeError(
            f"Refusing remote base URL: {base_url}. "
            "Use --allow-remote-base-url to override intentionally."
        )


def _codehub_headers(internal_api_key: str, bearer_token: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    safe_internal = str(internal_api_key or "").strip()
    safe_bearer = str(bearer_token or "").strip()
    if safe_bearer:
        headers["Authorization"] = f"Bearer {safe_bearer}"
    if safe_internal:
        headers["X-Internal-API-Key"] = safe_internal
    return headers


def _probe_codehub_quantum_path(
    base_url: str,
    timeout_sec: int,
    internal_api_key: str,
    bearer_token: str,
) -> dict[str, Any]:
    safe_base = str(base_url or "").strip().rstrip("/")
    if not safe_base:
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "CODEHUB backend URL not configured",
        }

    headers = _codehub_headers(internal_api_key=internal_api_key, bearer_token=bearer_token)
    health_url = f"{safe_base}/health"
    generate_url = f"{safe_base}/api/code/generate"

    health_status = None
    health_error = ""
    try:
        health_response = requests.get(health_url, timeout=max(5, int(timeout_sec)))
        health_status = int(health_response.status_code)
    except Exception as exc:
        health_error = str(exc)

    generate_status = None
    generate_error = ""
    quantum_layer_present = False
    try:
        generate_response = requests.post(
            generate_url,
            headers=headers,
            json={
                "prompt": (
                    "Create a minimal qiskit module with class QuantumLayer and forward(self, x) "
                    "for a 2-qubit bell-like circuit."
                ),
                "framework": "qiskit",
                "num_qubits": 2,
                "include_explanation": False,
                "client_context": {"client_type": "api", "client_version": "research_platform_test"},
            },
            timeout=max(5, int(timeout_sec)),
        )
        generate_status = int(generate_response.status_code)
        body = _safe_json(generate_response)
        code = str((body.get("code") if isinstance(body, dict) else "") or "")
        quantum_layer_present = "class QuantumLayer" in code and "def forward(" in code
        if generate_status != 200:
            generate_error = str(body)[:500]
    except Exception as exc:
        generate_error = str(exc)

    passed = (health_status == 200) and (generate_status == 200) and quantum_layer_present
    return {
        "enabled": True,
        "status": "pass" if passed else "fail",
        "base_url": safe_base,
        "health_status": health_status,
        "health_error": health_error,
        "generate_status": generate_status,
        "generate_error": generate_error,
        "quantum_layer_present": quantum_layer_present,
        "auth": {
            "internal_api_key_configured": bool(str(internal_api_key or "").strip()),
            "bearer_token_configured": bool(str(bearer_token or "").strip()),
        },
    }


class ApiHarness:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.records: list[dict[str, Any]] = []

    def request(
        self,
        scenario: str,
        step: str,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout_sec: int = 90,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        started = time.time()
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=_headers(),
            json=body,
            timeout=timeout_sec,
        )
        elapsed_ms = (time.time() - started) * 1000.0
        parsed_body = _safe_json(response)
        record = {
            "scenario": scenario,
            "step": step,
            "request": {
                "method": method.upper(),
                "path": path,
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "body": body,
                "timestamp": started,
            },
            "response": {
                "status_code": int(response.status_code),
                "headers": dict(response.headers),
                "body": parsed_body,
                "elapsed_ms": elapsed_ms,
                "timestamp": time.time(),
            },
            "normalized_endpoint": _normalize_endpoint(url),
        }
        self.records.append(record)
        return record


def _parse_action_commands(pending_action: dict[str, Any]) -> list[list[str]]:
    commands_value = pending_action.get("commands")
    if isinstance(commands_value, list) and commands_value:
        if all(isinstance(item, str) for item in commands_value):
            return [[str(item) for item in commands_value if str(item).strip()]]
        out: list[list[str]] = []
        for row in commands_value:
            if isinstance(row, list):
                cmd = [str(item) for item in row if str(item).strip()]
                if cmd:
                    out.append(cmd)
        if out:
            return out

    command_value = pending_action.get("command")
    if isinstance(command_value, list):
        cmd = [str(item) for item in command_value if str(item).strip()]
        return [cmd] if cmd else []
    return []


def _write_file_operations(file_operations: list[dict[str, Any]]) -> list[str]:
    created: list[str] = []
    for item in file_operations:
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        content = str(item.get("content", ""))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        created.append(path)
    return created


def _run_command(command: list[str], cwd: str, timeout_seconds: int) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=cwd or None,
            timeout=max(1, int(timeout_seconds or 600)),
        )
        return {
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout),
            "stderr": str(proc.stderr),
            "duration_sec": float(time.time() - started),
            "command": command,
            "cwd": cwd,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": str(exc.stdout or ""),
            "stderr": "TimeoutExpired",
            "duration_sec": float(time.time() - started),
            "command": command,
            "cwd": cwd,
        }


def _execute_pending_action(
    pending_action: dict[str, Any],
    inject_one_failure_for_run: bool,
    install_actions_execute: bool,
    failure_state: dict[str, bool],
) -> dict[str, Any]:
    action = str(pending_action.get("action", "")).strip().lower()
    cwd = str(pending_action.get("cwd", "")).strip()
    timeout_seconds = int(pending_action.get("timeout_seconds") or 600)
    file_operations = pending_action.get("file_operations", [])
    created_files = _write_file_operations(file_operations if isinstance(file_operations, list) else [])
    commands = _parse_action_commands(pending_action)

    if action == "install_package" and not install_actions_execute:
        command = commands[0] if commands else ["python", "-m", "pip", "install", "simulated"]
        return {
            "returncode": 0,
            "stdout": "Simulated package install success",
            "stderr": "",
            "duration_sec": 0.01,
            "command": command,
            "cwd": cwd,
            "created_files": created_files,
            "metadata": {"simulated": True},
        }

    if action == "run_local_commands" and inject_one_failure_for_run and not failure_state.get("failed_once", False):
        failure_state["failed_once"] = True
        command = commands[0] if commands else ["python", "main.py"]
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": "RuntimeError: simulated local execution failure",
            "duration_sec": 0.01,
            "command": command,
            "cwd": cwd,
            "created_files": created_files,
            "metadata": {"simulated_failure": True},
        }

    if not commands:
        return {
            "returncode": 0,
            "stdout": "No command execution required",
            "stderr": "",
            "duration_sec": 0.0,
            "command": [],
            "cwd": cwd,
            "created_files": created_files,
            "metadata": {},
        }

    aggregate_stdout: list[str] = []
    aggregate_stderr: list[str] = []
    total_duration = 0.0
    last_returncode = 0
    last_command: list[str] = commands[-1]
    for command in commands:
        result = _run_command(command=command, cwd=cwd, timeout_seconds=timeout_seconds)
        last_command = result["command"]
        total_duration += float(result["duration_sec"])
        last_returncode = int(result["returncode"])
        aggregate_stdout.append(str(result["stdout"]))
        aggregate_stderr.append(str(result["stderr"]))
        if last_returncode != 0:
            break
    return {
        "returncode": last_returncode,
        "stdout": "\n".join([part for part in aggregate_stdout if part]),
        "stderr": "\n".join([part for part in aggregate_stderr if part]),
        "duration_sec": total_duration,
        "command": last_command,
        "cwd": cwd,
        "created_files": created_files,
        "metadata": {},
    }


def _answer_clarifications(harness: ApiHarness, scenario: str, experiment_id: str, initial_pending: dict[str, Any]) -> None:
    pending = initial_pending or {}
    while pending:
        current = pending.get("current_question")
        if not isinstance(current, dict):
            questions = pending.get("questions") or []
            current = questions[0] if questions else None
        if not isinstance(current, dict) or not current.get("id"):
            return
        answer = _default_answer(current)
        answer_payload = {"answers": {str(current["id"]): answer}}
        response = harness.request(
            scenario=scenario,
            step=f"answer_{current['id']}",
            method="POST",
            path=f"/research/{experiment_id}/answer",
            body=answer_payload,
        )
        pending = _extract_data(response["response"]["body"]).get("pending_questions") or {}


def _collect_working_agents(log_payload: dict[str, Any]) -> list[str]:
    working: set[str] = set()
    logs = (log_payload.get("logs") or []) if isinstance(log_payload, dict) else []
    for row in logs:
        phase = str(row.get("phase") or "").strip()
        if phase and phase not in {"system", "unknown", "finished"}:
            working.add(phase)
        message = str(row.get("message") or "")
        if message.startswith("Phase ") and (" started" in message or " completed" in message):
            parts = message.split()
            if len(parts) >= 2:
                candidate = str(parts[1]).strip()
                if candidate:
                    working.add(candidate)
        if "Clarification" in message:
            working.add("clarifier")
    if "finished" in working:
        working.remove("finished")
    return sorted(working)


def _run_research_scenario(
    harness: ApiHarness,
    scenario: str,
    prompt: str,
    research_type: str,
    timeout_sec: int,
    poll_interval: float,
    inject_one_failure_for_run: bool,
    install_actions_execute: bool,
) -> dict[str, Any]:
    start_payload = {
        "prompt": prompt,
        "research_type": research_type,
        "priority": "normal",
        "tags": [scenario, "full-agent-test"],
        "user_id": f"test_{scenario}",
        "test_mode": True,
        "config_overrides": {"random_seed": 42, "hardware_target": "cpu", "max_epochs": 20, "output_format": ".py"},
    }
    start = harness.request(scenario, "start", "POST", "/research/start", start_payload)
    start_data = _extract_data(start["response"]["body"])
    experiment_id = str(start_data.get("experiment_id", ""))
    if not experiment_id:
        raise RuntimeError(f"Failed to start scenario {scenario}")

    _answer_clarifications(
        harness=harness,
        scenario=scenario,
        experiment_id=experiment_id,
        initial_pending=start_data.get("pending_questions") or {},
    )

    failure_state = {"failed_once": False}
    deadline = time.time() + timeout_sec
    latest_status: dict[str, Any] = {}
    while time.time() < deadline:
        status_resp = harness.request(scenario, "status_poll", "GET", f"/research/{experiment_id}/status")
        latest_status = _extract_data(status_resp["response"]["body"])
        status_value = str(latest_status.get("status", ""))
        if status_value in {"success", "failed", "aborted"}:
            break

        if bool(latest_status.get("waiting_for_user")):
            pending_action = latest_status.get("pending_action") or {}
            action_id = str(pending_action.get("action_id", ""))
            if action_id:
                execution_result = _execute_pending_action(
                    pending_action=pending_action,
                    inject_one_failure_for_run=inject_one_failure_for_run,
                    install_actions_execute=install_actions_execute,
                    failure_state=failure_state,
                )
                confirm_payload = {
                    "action_id": action_id,
                    "decision": "confirm",
                    "reason": f"Automated local action for scenario {scenario}",
                    "alternative_preference": "",
                    "execution_result": execution_result,
                }
                harness.request(
                    scenario=scenario,
                    step=f"confirm_{pending_action.get('action', 'action')}",
                    method="POST",
                    path=f"/research/{experiment_id}/confirm",
                    body=confirm_payload,
                )
        time.sleep(poll_interval)

    detail = harness.request(scenario, "research_get", "GET", f"/research/{experiment_id}")
    logs = harness.request(scenario, "logs", "GET", f"/research/{experiment_id}/logs?limit=500&offset=0")
    results = harness.request(scenario, "results", "GET", f"/research/{experiment_id}/results")
    report = harness.request(scenario, "report", "GET", f"/research/{experiment_id}/report?format=markdown&download=false")
    files = harness.request(scenario, "files", "GET", f"/research/{experiment_id}/files")
    harness.request(scenario, "research_list", "GET", "/research?limit=20&offset=0")
    system_health = harness.request(scenario, "system_health", "GET", "/system/health")
    system_metrics = harness.request(scenario, "system_metrics", "GET", "/system/metrics")

    logs_data = _extract_data(logs["response"]["body"])
    detail_data = _extract_data(detail["response"]["body"])
    results_data = _extract_data(results["response"]["body"])
    working_agents = _collect_working_agents(logs_data)
    report_data = _extract_data(report["response"]["body"])
    file_data = _extract_data(files["response"]["body"])

    return {
        "scenario": scenario,
        "experiment_id": experiment_id,
        "final_status": str(latest_status.get("status", "")),
        "final_phase": str(latest_status.get("phase", "")),
        "agents_seen": working_agents,
        "llm_calls_count": int(detail_data.get("llm_calls_count", 0) or 0),
        "total_tokens_used": int(detail_data.get("total_tokens_used", 0) or 0),
        "results": results_data,
        "report": {
            "path": report_data.get("report_path"),
            "word_count": report_data.get("word_count"),
            "sections": report_data.get("sections"),
        },
        "files": {
            "total_files": file_data.get("total_files"),
            "total_size_bytes": file_data.get("total_size_bytes"),
        },
        "system": {
            "health": _extract_data(system_health["response"]["body"]),
            "metrics": _extract_data(system_metrics["response"]["body"]),
        },
        "execution_logs": logs_data.get("execution_logs", []),
    }


def _run_abort_scenario(harness: ApiHarness, timeout_sec: int) -> dict[str, Any]:
    start_payload = {
        "prompt": "Create an experiment and abort immediately for abort path validation",
        "priority": "low",
        "tags": ["abort", "full-agent-test"],
        "user_id": "test_abort",
        "test_mode": True,
        "config_overrides": {"random_seed": 7, "hardware_target": "cpu"},
    }
    start = harness.request("abort", "start", "POST", "/research/start", start_payload, timeout_sec=timeout_sec)
    experiment_id = str(_extract_data(start["response"]["body"]).get("experiment_id", ""))
    if not experiment_id:
        raise RuntimeError("Abort scenario failed to start")
    abort_resp = harness.request(
        "abort",
        "abort",
        "DELETE",
        f"/research/{experiment_id}/abort",
        {"reason": "Automated abort scenario", "save_partial": True},
        timeout_sec=timeout_sec,
    )
    status = harness.request("abort", "status", "GET", f"/research/{experiment_id}/status", timeout_sec=timeout_sec)
    detail = harness.request("abort", "research_get", "GET", f"/research/{experiment_id}", timeout_sec=timeout_sec)
    logs = harness.request("abort", "logs", "GET", f"/research/{experiment_id}/logs?limit=500&offset=0", timeout_sec=timeout_sec)
    agents_seen = _collect_working_agents(_extract_data(logs["response"]["body"]))
    detail_data = _extract_data(detail["response"]["body"])
    return {
        "scenario": "abort",
        "experiment_id": experiment_id,
        "final_status": str(_extract_data(status["response"]["body"]).get("status", "")),
        "final_phase": str(_extract_data(status["response"]["body"]).get("phase", "")),
        "agents_seen": agents_seen,
        "llm_calls_count": int(detail_data.get("llm_calls_count", 0) or 0),
        "total_tokens_used": int(detail_data.get("total_tokens_used", 0) or 0),
        "abort_response": _extract_data(abort_resp["response"]["body"]),
    }


def _endpoint_coverage(records: list[dict[str, Any]]) -> list[str]:
    endpoints = sorted({f"{r['request']['method']} {r['normalized_endpoint']}" for r in records})
    return endpoints


def _validate_rl_feedback(scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    latest_metrics: dict[str, Any] = {}
    source_scenario = ""
    for item in reversed(scenario_results):
        metrics = ((item.get("system") or {}).get("metrics") or {})
        if isinstance(metrics, dict) and metrics:
            latest_metrics = metrics
            source_scenario = str(item.get("scenario", ""))
            break

    if not latest_metrics:
        return {
            "status": "fail",
            "reason": "system metrics missing from scenario results",
            "records": 0,
            "source_scenario": source_scenario,
        }

    rl = latest_metrics.get("rl_feedback", {}) if isinstance(latest_metrics, dict) else {}
    if not isinstance(rl, dict):
        return {
            "status": "fail",
            "reason": "rl_feedback missing in /system/metrics payload",
            "records": 0,
            "source_scenario": source_scenario,
        }

    records = int(rl.get("records", 0) or 0)
    by_phase_signal = rl.get("by_phase_signal", {})
    reward_trend = rl.get("reward_trend", [])
    has_distribution = isinstance(by_phase_signal, dict) and len(by_phase_signal) > 0
    has_trend = isinstance(reward_trend, list) and len(reward_trend) > 0
    passed = records > 0 and (has_distribution or has_trend)
    return {
        "status": "pass" if passed else "fail",
        "reason": "ok" if passed else "rl feedback records were empty or malformed",
        "records": records,
        "phase_signal_count": len(by_phase_signal) if isinstance(by_phase_signal, dict) else 0,
        "trend_points": len(reward_trend) if isinstance(reward_trend, list) else 0,
        "source_scenario": source_scenario,
    }


def _validate_llm_call_depth(scenario_results: list[dict[str, Any]], min_avg_calls: float) -> dict[str, Any]:
    per_run_calls: list[float] = []
    per_run_by_scenario: dict[str, float] = {}
    for item in scenario_results:
        calls_raw = item.get("llm_calls_count")
        if isinstance(calls_raw, (int, float)):
            calls = float(calls_raw)
            if calls >= 0:
                scenario_name = str(item.get("scenario", "unknown"))
                per_run_calls.append(calls)
                per_run_by_scenario[scenario_name] = calls

    if per_run_calls:
        avg_calls = float(sum(per_run_calls) / len(per_run_calls))
        passed = avg_calls >= float(min_avg_calls)
        return {
            "status": "pass" if passed else "fail",
            "reason": "ok" if passed else "avg_calls_per_experiment below configured minimum",
            "avg_calls_per_experiment": float(avg_calls),
            "minimum_required": float(min_avg_calls),
            "source": "scenario_results",
            "experiments_counted": len(per_run_calls),
            "per_experiment_calls": per_run_by_scenario,
        }

    latest_metrics: dict[str, Any] = {}
    source_scenario = ""
    for item in reversed(scenario_results):
        metrics = ((item.get("system") or {}).get("metrics") or {})
        if isinstance(metrics, dict) and metrics:
            latest_metrics = metrics
            source_scenario = str(item.get("scenario", ""))
            break

    if not latest_metrics:
        return {
            "status": "fail",
            "reason": "system metrics missing from scenario results",
            "avg_calls_per_experiment": 0.0,
            "minimum_required": float(min_avg_calls),
            "source": "none",
            "source_scenario": source_scenario,
        }

    llm_usage = latest_metrics.get("llm_usage", {})
    avg_calls = float(llm_usage.get("avg_calls_per_experiment", 0.0) or 0.0) if isinstance(llm_usage, dict) else 0.0
    passed = avg_calls >= float(min_avg_calls)
    return {
        "status": "pass" if passed else "fail",
        "reason": "ok" if passed else "avg_calls_per_experiment below configured minimum",
        "avg_calls_per_experiment": float(avg_calls),
        "minimum_required": float(min_avg_calls),
        "source": "system_metrics",
        "source_scenario": source_scenario,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-JSON end-to-end tester covering all research agents in VS-first mode.")
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1", help="API base URL")
    parser.add_argument(
        "--output-json",
        default=f"./workspace/test_outputs/{_now_tag()}_full_run.json",
        help="Single JSON output path",
    )
    parser.add_argument(
        "--scenarios",
        default="standard,quantum,recovery,abort",
        help="Comma-separated scenarios: standard,quantum,recovery,abort",
    )
    parser.add_argument(
        "--ai-only",
        action="store_true",
        help="Run AI-only scenarios (standard,recovery,abort) and skip quantum scenario.",
    )
    parser.add_argument("--poll-interval", type=float, default=1.5, help="Polling interval in seconds")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per-scenario timeout in seconds")
    parser.add_argument(
        "--require-rl-feedback",
        action="store_true",
        help="Fail run when RL feedback records are missing or empty in /system/metrics.",
    )
    parser.add_argument(
        "--min-avg-llm-calls",
        type=float,
        default=2.5,
        help="Minimum acceptable avg LLM calls per experiment (scenario result first, /system/metrics fallback).",
    )
    parser.add_argument(
        "--execute-install-actions",
        action="store_true",
        help="Actually run pip install actions from pending local actions (off by default, simulated success).",
    )
    parser.add_argument(
        "--allow-remote-base-url",
        action="store_true",
        help="Allow non-localhost base URL (disabled by default).",
    )
    parser.add_argument(
        "--codehub-base-url",
        default=str(os.getenv("CODEHUB_BACKEND_BASE_URL", "")).strip(),
        help="Optional CodeHub backend base URL (e.g. http://127.0.0.1:8001).",
    )
    parser.add_argument(
        "--codehub-timeout-sec",
        type=int,
        default=45,
        help="Timeout for CodeHub integration checks.",
    )
    parser.add_argument(
        "--codehub-internal-api-key",
        default=str(os.getenv("CODEHUB_INTERNAL_API_KEY", "")).strip(),
        help="Optional internal API key sent as X-Internal-API-Key to CodeHub.",
    )
    parser.add_argument(
        "--codehub-bearer-token",
        default=str(os.getenv("CODEHUB_BEARER_TOKEN", "")).strip(),
        help="Optional bearer token sent as Authorization to CodeHub.",
    )
    args = parser.parse_args()

    _enforce_local_base_url(args.base_url, allow_remote=args.allow_remote_base_url)
    selected = [part.strip().lower() for part in str(args.scenarios).split(",") if part.strip()]
    if args.ai_only:
        selected = [name for name in selected if name in {"standard", "recovery", "abort"}]
        if not selected:
            selected = ["standard", "recovery", "abort"]
    allowed = {"standard", "quantum", "recovery", "abort"}
    invalid = sorted([name for name in selected if name not in allowed])
    if invalid:
        print(f"ERROR: invalid scenarios: {invalid}", file=sys.stderr)
        return 2

    harness = ApiHarness(base_url=args.base_url)
    scenario_results: list[dict[str, Any]] = []
    scenario_errors: list[dict[str, str]] = []

    for scenario in selected:
        try:
            if scenario == "standard":
                scenario_results.append(
                    _run_research_scenario(
                        harness=harness,
                        scenario="standard",
                        prompt="Build a low-complexity classical classifier and evaluate accuracy on synthetic data",
                        research_type="ai",
                        timeout_sec=args.timeout_sec,
                        poll_interval=args.poll_interval,
                        inject_one_failure_for_run=False,
                        install_actions_execute=args.execute_install_actions,
                    )
                )
            elif scenario == "quantum":
                scenario_results.append(
                    _run_research_scenario(
                        harness=harness,
                        scenario="quantum",
                        prompt="Build an advanced hybrid quantum-classical pipeline with qiskit circuit integration and fidelity tracking",
                        research_type="quantum",
                        timeout_sec=args.timeout_sec,
                        poll_interval=args.poll_interval,
                        inject_one_failure_for_run=False,
                        install_actions_execute=args.execute_install_actions,
                    )
                )
            elif scenario == "recovery":
                scenario_results.append(
                    _run_research_scenario(
                        harness=harness,
                        scenario="recovery",
                        prompt="Run intermediate-level local training and recover automatically from one execution failure",
                        research_type="ai",
                        timeout_sec=args.timeout_sec,
                        poll_interval=args.poll_interval,
                        inject_one_failure_for_run=True,
                        install_actions_execute=args.execute_install_actions,
                    )
                )
            elif scenario == "abort":
                scenario_results.append(_run_abort_scenario(harness=harness, timeout_sec=args.timeout_sec))
        except Exception as exc:
            scenario_errors.append({"scenario": scenario, "error": str(exc)})

    covered_agents = sorted({agent for item in scenario_results for agent in item.get("agents_seen", [])})
    missing_agents = sorted([agent for agent in EXPECTED_AGENTS if agent not in covered_agents])
    latency_values = [float(record["response"]["elapsed_ms"]) for record in harness.records]
    endpoint_coverage = _endpoint_coverage(harness.records)
    quantum_selected = "quantum" in selected
    codehub_probe = _probe_codehub_quantum_path(
        base_url=args.codehub_base_url,
        timeout_sec=args.codehub_timeout_sec,
        internal_api_key=args.codehub_internal_api_key,
        bearer_token=args.codehub_bearer_token,
    )
    rl_validation = _validate_rl_feedback(scenario_results)
    llm_call_depth = _validate_llm_call_depth(scenario_results, min_avg_calls=float(args.min_avg_llm_calls))
    codehub_gate_ok = True
    if quantum_selected:
        codehub_gate_ok = codehub_probe.get("status", "pass") in {"pass", "skipped"}
    rl_gate_ok = (rl_validation.get("status") == "pass") if bool(args.require_rl_feedback) else True
    llm_gate_ok = llm_call_depth.get("status") == "pass"

    output = {
        "generated_at": time.time(),
        "base_url": args.base_url,
        "selected_scenarios": selected,
        "mode": "ai_only" if args.ai_only else "mixed",
        "execute_install_actions": bool(args.execute_install_actions),
        "require_rl_feedback": bool(args.require_rl_feedback),
        "min_avg_llm_calls": float(args.min_avg_llm_calls),
        "requests_total": len(harness.records),
        "latency": _latency_stats(latency_values),
        "endpoint_coverage": endpoint_coverage,
        "expected_agents": EXPECTED_AGENTS,
        "covered_agents": covered_agents,
        "missing_agents": missing_agents,
        "scenario_results": scenario_results,
        "scenario_errors": scenario_errors,
        "external_integrations": {
            "codehub_quantum": codehub_probe,
            "rl_feedback_validation": rl_validation,
            "llm_call_depth_validation": llm_call_depth,
        },
        "requests": harness.records,
        "result": "success"
        if (not scenario_errors and codehub_gate_ok and rl_gate_ok and llm_gate_ok)
        else "partial_failure",
    }

    output_path = Path(args.output_json).resolve()
    _save_json(output_path, output)
    print(f"Done. Single JSON report written to: {output_path}")
    return 0 if not scenario_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
