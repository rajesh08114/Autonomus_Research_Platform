from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import requests

EXPERIMENT_ID_RE = re.compile(r"exp_\d{8}_[0-9a-f]+", re.IGNORECASE)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
KNOWN_ENDPOINTS: set[tuple[str, str]] = {
    ("POST", "/research/start"),
    ("POST", "/research/{experiment_id}/answer"),
    ("POST", "/research/{experiment_id}/confirm"),
    ("GET", "/research/{experiment_id}"),
    ("GET", "/research/{experiment_id}/status"),
    ("GET", "/research/{experiment_id}/logs?limit&offset"),
    ("GET", "/research/{experiment_id}/results"),
    ("GET", "/research/{experiment_id}/report?download&format"),
    ("GET", "/research?limit&offset"),
    ("DELETE", "/research/{experiment_id}/abort"),
    ("POST", "/research/{experiment_id}/retry"),
    ("GET", "/research/{experiment_id}/files"),
    ("GET", "/research/{experiment_id}/files/{file_path}"),
    ("GET", "/research/{experiment_id}/plots/{plot_name}"),
    ("GET", "/system/health"),
    ("GET", "/system/metrics"),
}


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_json(target: Path, payload: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def _redacted_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_data(envelope: dict[str, Any]) -> dict[str, Any]:
    body = envelope.get("body", {})
    if isinstance(body, dict):
        return body.get("data", {}) if "data" in body else body
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
        query_keys = sorted({key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)})
        if query_keys:
            return f"{path}?{'&'.join(query_keys)}"
    return path


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))
    return {
        "count": float(len(ordered)),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "avg_ms": sum(ordered) / len(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_idx],
    }


def _default_answer(question: dict[str, Any]) -> Any:
    qid = question.get("id")
    qtype = question.get("type")
    options = question.get("options") or []
    default = question.get("default")
    if default is not None:
        return default
    if qid == "Q3":
        return True
    if qid == "Q4":
        return "pennylane"
    if qid == "Q5":
        return "synthetic"
    if qid == "Q7":
        return "accuracy"
    if qid == "Q8":
        return "cpu"
    if qid == "Q10":
        return 42
    if qid == "Q11":
        return 20
    if qtype == "boolean":
        return False
    if qtype == "number":
        return 1
    if qtype == "choice" and options:
        return options[0]
    return "auto"


def _enforce_local_base_url(base_url: str, allow_remote: bool) -> None:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if not allow_remote and host not in LOCAL_HOSTS:
        raise RuntimeError(
            f"Refusing remote base URL ({base_url}). "
            "This tester is configured for user-local execution only. "
            "Use --allow-remote-base-url to override intentionally."
        )


class ScenarioRunner:
    def __init__(self, scenario: str, base_url: str, output_dir: Path) -> None:
        self.scenario = scenario
        self.base_url = base_url.rstrip("/")
        self.output_dir = output_dir
        self.requests_dir = output_dir / "requests"
        self.request_index = 0
        self.records: list[dict[str, Any]] = []
        self.coverage: set[tuple[str, str]] = set()

    def request(self, name: str, method: str, path: str, body: dict | None = None, timeout_sec: int = 60) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        started = time.time()
        outbound_headers = _headers()
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=outbound_headers,
            json=body,
            timeout=timeout_sec,
        )
        elapsed_ms = (time.time() - started) * 1000.0
        headers = dict(response.headers)
        content_type = response.headers.get("content-type", "")

        parsed_body: Any
        if "application/json" in content_type:
            try:
                parsed_body = response.json()
            except Exception:
                parsed_body = {"raw_text": response.text[:10000]}
        elif "text/" in content_type:
            parsed_body = {"raw_text": response.text[:10000]}
        else:
            parsed_body = {"raw_bytes_len": len(response.content), "content_type": content_type}

        normalized = _normalize_endpoint(url)
        self.coverage.add((method.upper(), normalized))

        record = {
            "scenario": self.scenario,
            "name": name,
            "step_index": self.request_index,
            "normalized_endpoint": normalized,
            "request": {
                "method": method.upper(),
                "path": path,
                "url": url,
                "headers": _redacted_headers(),
                "body": body,
                "timestamp": started,
            },
            "response": {
                "status_code": response.status_code,
                "headers": headers,
                "body": parsed_body,
                "elapsed_ms": elapsed_ms,
                "timestamp": time.time(),
            },
            # Backward-compatible keys used by analyze_run.py.
            "url": url,
            "status_code": response.status_code,
            "headers": headers,
            "body": parsed_body,
            "elapsed_ms": elapsed_ms,
        }
        self.request_index += 1
        self.records.append(record)
        _save_json(self.requests_dir / f"{record['step_index']:03d}_{name}.json", record)
        return record


def _run_research_flow(
    runner: ScenarioRunner,
    start_body: dict[str, Any],
    poll_interval: float,
    timeout_sec: int,
    include_retry: bool,
    auto_confirm_decision: str = "confirm",
) -> dict[str, Any]:
    start = runner.request("start", "POST", "/research/start", start_body)
    start_data = _extract_data(start)
    experiment_id = start_data.get("experiment_id")
    if not experiment_id:
        raise RuntimeError(f"Failed to start experiment: {start}")

    runner.request("research_get_initial", "GET", f"/research/{experiment_id}")
    pending_payload = start_data.get("pending_questions") or {}
    while pending_payload:
        active = pending_payload.get("current_question")
        if not isinstance(active, dict):
            questions = pending_payload.get("questions") or []
            active = questions[0] if questions else None
        if not isinstance(active, dict) or not active.get("id"):
            break

        answer_payload = {"answers": {active["id"]: _default_answer(active)}}
        answer_resp = runner.request("answer", "POST", f"/research/{experiment_id}/answer", answer_payload)
        answer_data = _extract_data(answer_resp)
        pending_payload = answer_data.get("pending_questions") or {}
        if not pending_payload:
            break

    deadline = time.time() + timeout_sec
    latest_status = {}
    while time.time() < deadline:
        status = runner.request("status_poll", "GET", f"/research/{experiment_id}/status")
        status_data = _extract_data(status)
        latest_status = status_data
        value = status_data.get("status")
        if value in {"success", "failed", "aborted"}:
            break

        if status_data.get("waiting_for_user"):
            pending_action = status_data.get("pending_action") or {}
            action_id = pending_action.get("action_id")
            if action_id:
                confirm_payload = {
                    "action_id": action_id,
                    "decision": auto_confirm_decision,
                    "reason": f"Automated {runner.scenario} scenario confirmation",
                    "alternative_preference": "",
                }
                runner.request("confirm", "POST", f"/research/{experiment_id}/confirm", confirm_payload)
        time.sleep(poll_interval)

    # Post-run endpoint sweep.
    detail = runner.request("research_get_final", "GET", f"/research/{experiment_id}")
    logs = runner.request("logs", "GET", f"/research/{experiment_id}/logs?limit=500&offset=0")
    files = runner.request("files", "GET", f"/research/{experiment_id}/files")
    results = runner.request("results", "GET", f"/research/{experiment_id}/results")
    report = runner.request("report", "GET", f"/research/{experiment_id}/report?format=markdown&download=false")
    runner.request("research_list", "GET", "/research?limit=20&offset=0")
    health = runner.request("system_health", "GET", "/system/health")
    metrics = runner.request("system_metrics", "GET", "/system/metrics")

    files_data = _extract_data(files)
    file_items = files_data.get("files", []) if isinstance(files_data, dict) else []
    for idx, item in enumerate(file_items[:3], start=1):
        rel = item.get("path")
        if rel:
            runner.request(f"file_content_{idx}", "GET", f"/research/{experiment_id}/files/{rel}")

    # Endpoint coverage for plot route even when plot may not exist.
    runner.request("plot_probe", "GET", f"/research/{experiment_id}/plots/nonexistent.png")

    retry_response = None
    if include_retry:
        retry_payload = {
            "from_phase": "results_evaluator",
            "reset_retries": False,
            "override_config": {"hardware_target": "cpu"},
        }
        retry_response = runner.request("retry", "POST", f"/research/{experiment_id}/retry", retry_payload)
        retry_deadline = time.time() + min(timeout_sec, 180)
        while time.time() < retry_deadline:
            retry_status = runner.request("status_after_retry", "GET", f"/research/{experiment_id}/status")
            retry_data = _extract_data(retry_status)
            latest_status = retry_data
            if retry_data.get("status") in {"success", "failed", "aborted"}:
                break
            time.sleep(poll_interval)
        runner.request("results_after_retry", "GET", f"/research/{experiment_id}/results")

    return {
        "experiment_id": experiment_id,
        "final_status": latest_status.get("status"),
        "final_phase": latest_status.get("phase"),
        "detail": detail,
        "logs": logs,
        "results": results,
        "report": report,
        "health": health,
        "metrics": metrics,
        "retry": retry_response,
    }


def _run_abort_flow(runner: ScenarioRunner) -> dict[str, Any]:
    start_body = {
        "prompt": "Start an experiment and abort it for endpoint validation",
        "priority": "low",
        "tags": ["abort", "endpoint-test"],
        "config_overrides": {"random_seed": 7, "hardware_target": "cpu"},
    }
    start = runner.request("abort_start", "POST", "/research/start", start_body)
    exp_id = _extract_data(start).get("experiment_id")
    if not exp_id:
        raise RuntimeError(f"Abort scenario failed to start: {start}")

    abort_body = {"reason": "Automated abort scenario", "save_partial": True}
    abort = runner.request("abort", "DELETE", f"/research/{exp_id}/abort", abort_body)
    status = runner.request("status_after_abort", "GET", f"/research/{exp_id}/status")
    runner.request("research_get_abort", "GET", f"/research/{exp_id}")
    logs = runner.request("logs_abort", "GET", f"/research/{exp_id}/logs?limit=500&offset=0")
    runner.request("results_abort", "GET", f"/research/{exp_id}/results")
    runner.request("files_abort", "GET", f"/research/{exp_id}/files")
    runner.request("research_list_abort", "GET", "/research?limit=20&offset=0")
    health = runner.request("system_health_abort", "GET", "/system/health")
    metrics = runner.request("system_metrics_abort", "GET", "/system/metrics")

    status_data = _extract_data(status)
    return {
        "experiment_id": exp_id,
        "final_status": status_data.get("status"),
        "final_phase": status_data.get("phase"),
        "abort_response": abort,
        "logs": logs,
        "health": health,
        "metrics": metrics,
    }


def _phase_validation(log_response: dict[str, Any], scenario: str) -> dict[str, Any]:
    data = _extract_data(log_response)
    logs = data.get("logs", []) if isinstance(data, dict) else []
    phase_counts: Counter[str] = Counter()
    completed: set[str] = set()
    started: set[str] = set()
    seen_phase_labels: set[str] = set()
    for row in logs:
        phase = str(row.get("phase") or "unknown")
        message = str(row.get("message") or "")
        phase_counts[phase] += 1
        if phase not in {"system", "unknown"}:
            seen_phase_labels.add(phase)
        if phase == "clarifier" or message.lower().startswith("clarification"):
            seen_phase_labels.add("clarifier")
        if message.startswith("Phase ") and message.endswith(" completed"):
            parts = message.split()
            if len(parts) >= 2:
                completed.add(parts[1])
        if message.startswith("Phase ") and message.endswith(" started"):
            parts = message.split()
            if len(parts) >= 2:
                started.add(parts[1])

    working = sorted(started | completed | seen_phase_labels)
    expected = [
        "clarifier",
        "planner",
        "env_manager",
        "dataset_manager",
        "code_generator",
        "job_scheduler",
        "subprocess_runner",
        "results_evaluator",
        "doc_generator",
    ]
    if scenario == "quantum":
        expected.append("quantum_gate")
    missing = sorted([phase for phase in expected if phase not in working])
    return {
        "expected_agents": expected,
        "working_agents": working,
        "missing_agents": missing,
        "phase_log_counts": dict(phase_counts),
    }


def _scenario_metrics(scenario: str, runner: ScenarioRunner, flow: dict[str, Any]) -> dict[str, Any]:
    latencies = [float(r.get("elapsed_ms", 0.0)) for r in runner.records]
    status_codes = [int(r.get("status_code", 0)) for r in runner.records]
    ok = sum(1 for code in status_codes if 200 <= code < 300)
    warn = sum(1 for code in status_codes if 300 <= code < 400)
    err = sum(1 for code in status_codes if code >= 400)
    coverage = sorted([f"{m} {p}" for m, p in runner.coverage])

    detail_data = _extract_data(flow.get("detail", {}))
    health_data = _extract_data(flow.get("health", {}))
    llm = detail_data.get("llm", {}) if isinstance(detail_data, dict) else {}
    if not llm:
        llm = ((health_data.get("components") or {}).get("master_llm") or {}) if isinstance(health_data, dict) else {}

    agent_validation = _phase_validation(flow.get("logs", {}), scenario) if flow.get("logs") else {}
    system_metrics_payload = _extract_data(flow.get("metrics", {})) if flow.get("metrics") else {}

    return {
        "scenario": scenario,
        "experiment_id": flow.get("experiment_id"),
        "final_status": flow.get("final_status"),
        "final_phase": flow.get("final_phase"),
        "requests_total": len(runner.records),
        "requests_ok": ok,
        "requests_redirect": warn,
        "requests_error": err,
        "latency": _latency_stats(latencies),
        "covered_endpoints": coverage,
        "llm_used": {
            "provider": llm.get("provider"),
            "model": llm.get("model"),
            "status": llm.get("status"),
        },
        "agent_validation": agent_validation,
        "system_metrics_snapshot": system_metrics_payload,
    }


def _build_agent_trace(scenario: str, runner: ScenarioRunner, flow: dict[str, Any]) -> dict[str, Any]:
    questions_asked: list[dict[str, Any]] = []
    answers_sent: list[dict[str, Any]] = []
    confirmations_sent: list[dict[str, Any]] = []
    api_responses: list[dict[str, Any]] = []
    pending_actions: list[dict[str, Any]] = []

    for record in runner.records:
        name = str(record.get("name", ""))
        request_payload = (record.get("request") or {}).get("body")
        response_payload = _extract_data(record)
        status_code = int(record.get("status_code", 0))

        api_responses.append(
            {
                "step": name,
                "endpoint": record.get("normalized_endpoint"),
                "status_code": status_code,
                "elapsed_ms": record.get("elapsed_ms"),
            }
        )

        pending_q = (response_payload.get("pending_questions") or {}).get("questions", []) if isinstance(response_payload, dict) else []
        if pending_q:
            questions_asked.append(
                {
                    "step": name,
                    "endpoint": record.get("normalized_endpoint"),
                    "questions": pending_q,
                }
            )

        if name == "answer" and isinstance(request_payload, dict):
            answers_sent.append(
                {
                    "step": name,
                    "endpoint": record.get("normalized_endpoint"),
                    "answers": request_payload.get("answers", {}),
                }
            )
        if name == "confirm" and isinstance(request_payload, dict):
            confirmations_sent.append(
                {
                    "step": name,
                    "endpoint": record.get("normalized_endpoint"),
                    "confirmation": request_payload,
                }
            )

    logs_data = _extract_data(flow.get("logs", {})) if flow.get("logs") else {}
    logs = logs_data.get("logs", []) if isinstance(logs_data, dict) else []
    execution_logs = logs_data.get("execution_logs", []) if isinstance(logs_data, dict) else []

    command_events: list[dict[str, Any]] = []
    for row in logs:
        if str(row.get("message")) in {"Subprocess execution started", "Subprocess execution finished"}:
            command_events.append(
                {
                    "phase": row.get("phase"),
                    "level": row.get("level"),
                    "message": row.get("message"),
                    "details": row.get("details", {}),
                    "timestamp": row.get("timestamp"),
                }
            )
        if str(row.get("message")) == "User confirmation required":
            pending_actions.append(
                {
                    "phase": row.get("phase"),
                    "details": row.get("details", {}),
                    "timestamp": row.get("timestamp"),
                }
            )

    commands_from_execution_logs = []
    for item in execution_logs:
        commands_from_execution_logs.append(
            {
                "script_path": item.get("script_path"),
                "command": item.get("command"),
                "cwd": item.get("cwd"),
                "returncode": item.get("returncode"),
                "duration_sec": item.get("duration_sec"),
                "executor": item.get("executor"),
                "host": item.get("host"),
            }
        )

    detail_data = _extract_data(flow.get("detail", {}))
    health_data = _extract_data(flow.get("health", {}))
    llm = detail_data.get("llm", {}) if isinstance(detail_data, dict) else {}
    if not llm:
        llm = ((health_data.get("components") or {}).get("master_llm") or {}) if isinstance(health_data, dict) else {}

    return {
        "scenario": scenario,
        "experiment_id": flow.get("experiment_id"),
        "final_status": flow.get("final_status"),
        "final_phase": flow.get("final_phase"),
        "llm_used": {"provider": llm.get("provider"), "model": llm.get("model"), "status": llm.get("status")},
        "questions_asked": questions_asked,
        "answers_sent": answers_sent,
        "confirmations_sent": confirmations_sent,
        "pending_actions": pending_actions,
        "commands_from_logs": command_events,
        "commands_from_execution_logs": commands_from_execution_logs,
        "api_responses": api_responses,
    }


def _scenario_start_body(scenario: str) -> dict[str, Any]:
    if scenario == "quantum":
        return {
            "prompt": "Build a hybrid quantum-classical classifier using quantum circuits and evaluate on synthetic data",
            "priority": "high",
            "tags": ["quantum", "full-flow", "endpoint-test"],
            "config_overrides": {"random_seed": 42, "hardware_target": "cpu", "max_epochs": 20, "output_format": ".py"},
        }
    return {
        "prompt": "Build a classical classifier and evaluate accuracy on synthetic data",
        "priority": "normal",
        "tags": ["classical", "full-flow", "endpoint-test"],
        "config_overrides": {"random_seed": 42, "hardware_target": "cpu", "max_epochs": 20, "output_format": ".py"},
    }


def _run_selected_scenario(
    scenario: str,
    base_url: str,
    run_dir: Path,
    poll_interval: float,
    timeout_sec: int,
) -> dict[str, Any]:
    runner = ScenarioRunner(scenario=scenario, base_url=base_url, output_dir=run_dir)
    if scenario == "abort":
        flow = _run_abort_flow(runner)
    elif scenario == "retry":
        flow = _run_research_flow(
            runner=runner,
            start_body=_scenario_start_body("standard"),
            poll_interval=poll_interval,
            timeout_sec=timeout_sec,
            include_retry=True,
        )
    elif scenario in {"standard", "quantum"}:
        flow = _run_research_flow(
            runner=runner,
            start_body=_scenario_start_body(scenario),
            poll_interval=poll_interval,
            timeout_sec=timeout_sec,
            include_retry=False,
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    summary = _scenario_metrics(scenario, runner, flow)
    trace = _build_agent_trace(scenario, runner, flow)
    _save_json(run_dir / "90_flow.json", flow)
    _save_json(run_dir / "95_agent_trace.json", trace)
    _save_json(run_dir / "99_summary.json", summary)
    return {"flow": flow, "summary": summary, "trace": trace, "coverage": runner.coverage, "request_count": len(runner.records)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-scenario API endpoint tester with per-request JSON artifacts.")
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1", help="API base URL")
    parser.add_argument("--output-dir", default=f"./workspace/test_outputs/{_now_tag()}", help="Output directory root")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Status polling interval in seconds")
    parser.add_argument("--timeout-sec", type=int, default=600, help="Timeout per run scenario in seconds")
    parser.add_argument(
        "--scenarios",
        default="standard,quantum,retry,abort",
        help="Comma-separated scenarios: standard,quantum,retry,abort",
    )
    parser.add_argument(
        "--allow-remote-base-url",
        action="store_true",
        help="Allow non-localhost base URLs (disabled by default for local execution safety).",
    )
    args = parser.parse_args()

    _enforce_local_base_url(args.base_url, allow_remote=args.allow_remote_base_url)

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.time(),
        "base_url": args.base_url,
        "scenarios": args.scenarios,
        "local_only_guard": not args.allow_remote_base_url,
        "known_endpoints_count": len(KNOWN_ENDPOINTS),
    }
    _save_json(root / "00_manifest.json", manifest)

    selected = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]
    allowed = {"standard", "quantum", "retry", "abort"}
    invalid = [s for s in selected if s not in allowed]
    if invalid:
        print(f"ERROR: invalid scenarios: {invalid}", file=sys.stderr)
        return 2

    all_coverage: set[tuple[str, str]] = set()
    scenario_summaries: list[dict[str, Any]] = []
    scenario_errors: list[dict[str, str]] = []

    for scenario in selected:
        scenario_dir = root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = _run_selected_scenario(
                scenario=scenario,
                base_url=args.base_url,
                run_dir=scenario_dir,
                poll_interval=args.poll_interval,
                timeout_sec=args.timeout_sec,
            )
            scenario_summaries.append(result["summary"])
            all_coverage.update(result["coverage"])
        except Exception as exc:
            error_payload = {"scenario": scenario, "error": str(exc)}
            _save_json(scenario_dir / "99_error.json", error_payload)
            scenario_errors.append(error_payload)

    missing = sorted([f"{m} {p}" for m, p in (KNOWN_ENDPOINTS - all_coverage)])
    covered = sorted([f"{m} {p}" for m, p in all_coverage])
    endpoint_coverage = {
        "known_endpoints": sorted([f"{m} {p}" for m, p in KNOWN_ENDPOINTS]),
        "covered_endpoints": covered,
        "missing_endpoints": missing,
        "coverage_ratio": round((len(covered) / len(KNOWN_ENDPOINTS)) if KNOWN_ENDPOINTS else 1.0, 4),
    }
    _save_json(root / "01_endpoint_coverage.json", endpoint_coverage)

    final_summary = {
        "finished_at": time.time(),
        "base_url": args.base_url,
        "scenarios_selected": selected,
        "scenario_count": len(selected),
        "scenario_errors": scenario_errors,
        "scenario_summaries": scenario_summaries,
        "endpoint_coverage": endpoint_coverage,
        "result": "success" if not scenario_errors else "partial_failure",
        "output_dir": str(root),
    }
    _save_json(root / "99_summary.json", final_summary)
    print(f"Done. Test artifacts saved at: {root}")
    return 0 if not scenario_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
