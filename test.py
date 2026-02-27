from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import requests

EXPERIMENT_ID_RE = re.compile(r"exp_\d{8}_[0-9a-f]+", re.IGNORECASE)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
BASE_EXPECTED_AGENTS = [
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
SCENARIO_AGENT_REQUIREMENTS = {
    "quantum": "quantum_gate",
    "recovery": "error_recovery",
}


@dataclass(slots=True)
class ScenarioProfile:
    name: str
    research_type: str
    requires_quantum: bool
    dataset_source: str
    output_format: str
    target_metric: str
    code_level: str
    algorithm_class: str
    framework_preference: str
    hardware_target: str
    random_seed: int
    max_epochs: int
    quantum_framework: str
    quantum_algorithm: str
    quantum_qubit_count: int
    quantum_backend: str
    kaggle_dataset_id: str


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


def _profile_for(scenario: str, research_type: str, rng: random.Random) -> ScenarioProfile:
    quantum = research_type == "quantum" or scenario == "quantum"
    ai_metric = rng.choice(["accuracy", "f1", "roc_auc"])
    recovery_epochs = rng.randint(18, 36)
    standard_epochs = rng.randint(10, 24)
    quantum_epochs = rng.randint(24, 42)
    kaggle_choices = [
        "owner/dataset",
        "uciml/iris",
        "fedesoriano/heart-failure-prediction",
        "andrewmvd/heart-failure-clinical-data",
    ]
    return ScenarioProfile(
        name=scenario,
        research_type=research_type,
        requires_quantum=quantum,
        dataset_source="sklearn" if scenario == "standard" else rng.choice(["synthetic", "sklearn"]),
        output_format=".py",
        target_metric="fidelity" if quantum else ai_metric,
        code_level="advanced" if scenario == "quantum" else ("intermediate" if scenario == "recovery" else "low"),
        algorithm_class="quantum_ml" if quantum else "supervised",
        framework_preference="pytorch" if quantum else "sklearn",
        hardware_target="cpu",
        random_seed=rng.randint(1, 2_147_483_000),
        max_epochs=recovery_epochs if scenario == "recovery" else (quantum_epochs if scenario == "quantum" else standard_epochs),
        quantum_framework="qiskit",
        quantum_algorithm="QAOA",
        quantum_qubit_count=rng.randint(4, 8),
        quantum_backend=rng.choice(["aer_simulator", "statevector_simulator"]),
        kaggle_dataset_id=rng.choice(kaggle_choices),
    )


def _pick_choice(preferred: str, options: list[str]) -> str:
    if not options:
        return preferred
    lower_map = {str(item).strip().lower(): str(item) for item in options}
    key = str(preferred).strip().lower()
    if key in lower_map:
        return lower_map[key]
    for candidate in options:
        text = str(candidate).strip().lower()
        if key and key in text:
            return str(candidate)
    return str(options[0])


def _topic_from_question(question: dict[str, Any]) -> str:
    topic = str(question.get("topic", "")).strip().lower()
    if topic:
        return topic
    return str(question.get("text", "")).strip().lower()


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _bounded_int(value: int, low: int | None, high: int | None) -> int:
    out = int(value)
    if low is not None:
        out = max(out, int(low))
    if high is not None:
        out = min(out, int(high))
    return out


def _question_bounds(question: dict[str, Any]) -> tuple[int | None, int | None]:
    low = _int_or_none(question.get("minimum"))
    if low is None:
        low = _int_or_none(question.get("min"))
    high = _int_or_none(question.get("maximum"))
    if high is None:
        high = _int_or_none(question.get("max"))
    return low, high


def _dynamic_answer(
    question: dict[str, Any],
    profile: ScenarioProfile,
    answered_count: int,
    rng: random.Random,
    run_tag: str,
) -> Any:
    qtype = str(question.get("type", "text")).strip().lower()
    options = [str(x) for x in question.get("options", []) if str(x).strip()]
    default = question.get("default")
    signal = _topic_from_question(question)
    if default is not None and answered_count > 8 and rng.random() < 0.4:
        return default

    if qtype == "boolean":
        if "quantum" in signal:
            return bool(profile.requires_quantum)
        if "retry" in signal:
            return True
        if "gpu" in signal or "cuda" in signal:
            return False
        return bool(rng.random() < 0.75)

    if qtype == "number":
        low, high = _question_bounds(question)
        if "qubit" in signal:
            base = int(profile.quantum_qubit_count)
            return _bounded_int(base + rng.randint(-1, 1), low, high)
        if "epoch" in signal or "iteration" in signal:
            base = int(profile.max_epochs)
            jitter = max(1, int(base * 0.2))
            return _bounded_int(base + rng.randint(-jitter, jitter), low, high)
        if "seed" in signal:
            base = int(profile.random_seed)
            return _bounded_int(base + rng.randint(1, 999), low, high)
        if isinstance(default, int):
            return _bounded_int(default + rng.randint(-2, 2), low, high)
        return _bounded_int(rng.randint(3, 12), low, high)

    if qtype == "choice":
        preferred = "auto"
        if "output" in signal or "notebook" in signal:
            preferred = profile.output_format
        elif "algorithm" in signal and "quantum" not in signal:
            preferred = profile.algorithm_class
        elif "dataset" in signal:
            preferred = profile.dataset_source
        elif "metric" in signal:
            preferred = profile.target_metric
        elif "hardware" in signal or "gpu" in signal or "cuda" in signal:
            preferred = profile.hardware_target
        elif "framework" in signal and "quantum" not in signal:
            preferred = profile.framework_preference
        elif "code level" in signal or "complexity" in signal:
            preferred = profile.code_level
        elif "quantum framework" in signal:
            preferred = profile.quantum_framework
        elif "quantum algorithm" in signal:
            preferred = profile.quantum_algorithm
        elif "backend" in signal and "quantum" in signal:
            preferred = profile.quantum_backend
        elif "python" in signal:
            preferred = "3.11"
        elif "retry" in signal:
            preferred = "enabled"
        elif "problem" in signal:
            preferred = "classification"
        primary = _pick_choice(preferred, options)
        if len(options) <= 1:
            return primary
        if answered_count <= 1 or rng.random() < 0.7:
            return primary
        alternatives = [item for item in options if str(item) != str(primary)]
        if not alternatives:
            return primary
        return str(rng.choice(alternatives))

    if qtype == "text":
        if "kaggle" in signal:
            return profile.kaggle_dataset_id
        suffix = f"{run_tag}_{answered_count}_{rng.randint(100, 999)}"
        return f"{profile.name}_dynamic_{suffix}"

    if default is not None:
        return default
    if options:
        return options[0]
    return "auto"


def _expected_agents_for_selected(selected_scenarios: list[str], ensure_all_agents: bool) -> list[str]:
    expected = list(BASE_EXPECTED_AGENTS)
    if ensure_all_agents or "quantum" in selected_scenarios:
        expected.append(SCENARIO_AGENT_REQUIREMENTS["quantum"])
    if ensure_all_agents or "recovery" in selected_scenarios:
        expected.append(SCENARIO_AGENT_REQUIREMENTS["recovery"])
    return expected


def _extract_json_value(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw, count=1).strip()
        raw = re.sub(r"\s*```$", "", raw, count=1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        snippet = raw[first : last + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _coerce_llm_answer(question: dict[str, Any], llm_payload: Any) -> Any | None:
    value = llm_payload
    if isinstance(value, dict):
        if "answer" in value:
            value = value.get("answer")
    elif isinstance(value, str):
        parsed = _extract_json_value(value)
        if isinstance(parsed, dict) and "answer" in parsed:
            value = parsed.get("answer")
        elif parsed is not None:
            value = parsed

    qtype = str(question.get("type", "text")).strip().lower()
    options = [str(x) for x in question.get("options", []) if str(x).strip()]
    default = question.get("default")

    if qtype == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "on", "enabled"}:
            return True
        if text in {"false", "0", "no", "n", "off", "disabled"}:
            return False
        return None

    if qtype == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        text = str(value).strip()
        match = re.search(r"-?\d+", text)
        if match:
            try:
                return int(match.group(0))
            except Exception:
                return None
        return None

    if qtype == "choice":
        if not options:
            return str(value).strip() if value is not None else None
        text = str(value).strip()
        if text in options:
            return text
        lower_map = {str(opt).strip().lower(): str(opt) for opt in options}
        lowered = text.lower()
        if lowered in lower_map:
            return lower_map[lowered]
        for opt in options:
            if str(opt).strip().lower() in lowered:
                return str(opt)
        return str(default) if default in options else None

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _llm_clarification_answer(
    harness: "ApiHarness",
    scenario: str,
    question: dict[str, Any],
    profile: ScenarioProfile,
    run_tag: str,
    answered_count: int,
    timeout_sec: int,
) -> Any | None:
    qid = str(question.get("id", "")).strip() or "unknown"
    llm_prompt = (
        "You answer one clarification question for an autonomous research workflow.\n"
        "Return exactly one compact JSON object with key `answer` only.\n"
        "Rules:\n"
        "- For type=choice, answer must be one of the provided options exactly.\n"
        "- For type=boolean, answer must be true or false.\n"
        "- For type=number, answer must be an integer.\n"
        "- Do not include explanations.\n\n"
        f"Question JSON:\n{json.dumps(question, ensure_ascii=True)}\n\n"
        f"Scenario profile:\n{json.dumps(profile.__dict__, ensure_ascii=True)}\n\n"
        f"answered_count={answered_count}\n"
        "Respond now with JSON only."
    )
    payload = {
        "message": llm_prompt,
        "user_id": f"test_llm_{scenario}_{run_tag}",
        "test_mode": True,
        "context_limit": 1,
    }
    resp = harness.request(
        scenario=scenario,
        step=f"llm_answer_{qid}",
        method="POST",
        path="/chat/research",
        body=payload,
        timeout_sec=max(10, int(timeout_sec)),
    )
    status_code = int(resp["response"]["status_code"])
    if status_code >= 300:
        return None
    answer_blob = _extract_data(resp["response"]["body"]).get("answer")
    return _coerce_llm_answer(question, answer_blob)


def _enforce_local_base_url(base_url: str, allow_remote: bool) -> None:
    host = str(urlparse(base_url).hostname or "").lower()
    if not allow_remote and host not in LOCAL_HOSTS:
        raise RuntimeError(
            f"Refusing remote base URL: {base_url}. "
            "Use --allow-remote-base-url to override intentionally."
        )


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


def _answer_clarifications(
    harness: ApiHarness,
    scenario: str,
    experiment_id: str,
    initial_pending: dict[str, Any],
    profile: ScenarioProfile,
    rng: random.Random,
    run_tag: str,
    answer_mode: str,
    llm_answer_timeout_sec: int,
    answer_stats: dict[str, int] | None = None,
) -> int:
    mode = str(answer_mode).strip().lower() or "llm"
    pending = initial_pending or {}
    answered_count = 0
    while pending:
        current = pending.get("current_question")
        if not isinstance(current, dict):
            questions = pending.get("questions") or []
            current = questions[0] if questions else None
        if not isinstance(current, dict) or not current.get("id"):
            break
        answer = None
        if mode in {"llm", "llm_strict"}:
            if isinstance(answer_stats, dict):
                answer_stats["llm_attempts"] = int(answer_stats.get("llm_attempts", 0)) + 1
            try:
                answer = _llm_clarification_answer(
                    harness=harness,
                    scenario=scenario,
                    question=current,
                    profile=profile,
                    run_tag=run_tag,
                    answered_count=answered_count,
                    timeout_sec=llm_answer_timeout_sec,
                )
            except Exception:
                answer = None
            if answer is not None and isinstance(answer_stats, dict):
                answer_stats["llm_success"] = int(answer_stats.get("llm_success", 0)) + 1
        if answer is None:
            if mode == "llm_strict":
                raise RuntimeError(f"LLM failed to answer clarification question {current.get('id')}")
            answer = _dynamic_answer(
                current,
                profile=profile,
                answered_count=answered_count,
                rng=rng,
                run_tag=run_tag,
            )
            if isinstance(answer_stats, dict):
                answer_stats["fallback_used"] = int(answer_stats.get("fallback_used", 0)) + 1
        answer_payload = {"answers": {str(current["id"]): answer}}
        response = harness.request(
            scenario=scenario,
            step=f"answer_{current['id']}",
            method="POST",
            path=f"/research/{experiment_id}/answer",
            body=answer_payload,
        )
        if int(response["response"]["status_code"]) >= 300:
            raise RuntimeError(f"Answer failed for {current['id']}: {response['response']['body']}")
        pending = _extract_data(response["response"]["body"]).get("pending_questions") or {}
        answered_count += 1
    return answered_count


def _collect_working_agents(
    log_payload: dict[str, Any],
    phase_timings: dict[str, Any] | None = None,
    final_phase: str | None = None,
) -> list[str]:
    working: set[str] = set()
    logs = (log_payload.get("logs") or []) if isinstance(log_payload, dict) else []
    for row in logs:
        phase = str(row.get("phase") or "").strip()
        if phase and phase not in {"system", "unknown", "finished", "aborted"}:
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
    if isinstance(phase_timings, dict):
        for phase_name in phase_timings.keys():
            phase = str(phase_name or "").strip()
            if phase and phase not in {"system", "unknown", "finished", "aborted"}:
                working.add(phase)
    phase_value = str(final_phase or "").strip()
    if phase_value and phase_value not in {"system", "unknown", "finished", "aborted"}:
        working.add(phase_value)
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
    rng: random.Random,
    run_tag: str,
    answer_mode: str,
    llm_answer_timeout_sec: int,
) -> dict[str, Any]:
    profile = _profile_for(scenario, research_type, rng=rng)
    start_payload = {
        "prompt": prompt,
        "research_type": research_type,
        "priority": "normal",
        "tags": [scenario, "full-agent-test"],
        "user_id": f"test_{scenario}",
        "test_mode": True,
        "config_overrides": {
            "random_seed": profile.random_seed,
            "hardware_target": profile.hardware_target,
            "max_epochs": profile.max_epochs,
            "output_format": profile.output_format,
        },
    }
    start = harness.request(scenario, "start", "POST", "/research/start", start_payload)
    if int(start["response"]["status_code"]) != 201:
        raise RuntimeError(f"Failed to start scenario {scenario}: {start['response']['body']}")
    start_data = _extract_data(start["response"]["body"])
    experiment_id = str(start_data.get("experiment_id", ""))
    if not experiment_id:
        raise RuntimeError(f"Failed to start scenario {scenario}")

    answer_stats = {"llm_attempts": 0, "llm_success": 0, "fallback_used": 0}
    dynamic_answers_submitted = _answer_clarifications(
        harness=harness,
        scenario=scenario,
        experiment_id=experiment_id,
        initial_pending=start_data.get("pending_questions") or {},
        profile=profile,
        rng=rng,
        run_tag=run_tag,
        answer_mode=answer_mode,
        llm_answer_timeout_sec=llm_answer_timeout_sec,
        answer_stats=answer_stats,
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
            else:
                detail_waiting = harness.request(scenario, "research_get_waiting", "GET", f"/research/{experiment_id}")
                pending_questions = _extract_data(detail_waiting["response"]["body"]).get("pending_questions") or {}
                dynamic_answers_submitted += _answer_clarifications(
                    harness=harness,
                    scenario=scenario,
                    experiment_id=experiment_id,
                    initial_pending=pending_questions,
                    profile=profile,
                    rng=rng,
                    run_tag=run_tag,
                    answer_mode=answer_mode,
                    llm_answer_timeout_sec=llm_answer_timeout_sec,
                    answer_stats=answer_stats,
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
    working_agents = _collect_working_agents(
        logs_data,
        phase_timings=(detail_data.get("phase_timings") if isinstance(detail_data, dict) else None),
        final_phase=str(latest_status.get("phase", "")),
    )
    report_data = _extract_data(report["response"]["body"])
    file_data = _extract_data(files["response"]["body"])

    return {
        "scenario": scenario,
        "experiment_id": experiment_id,
        "dynamic_answers_submitted": dynamic_answers_submitted,
        "answering_stats": answer_stats,
        "profile": profile.__dict__,
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


def _run_abort_scenario(
    harness: ApiHarness,
    timeout_sec: int,
    rng: random.Random,
    run_tag: str,
    answer_mode: str,
    llm_answer_timeout_sec: int,
) -> dict[str, Any]:
    profile = _profile_for("abort", "ai", rng=rng)
    start_payload = {
        "prompt": "Create an experiment and abort immediately for abort path validation",
        "priority": "low",
        "tags": ["abort", "full-agent-test"],
        "user_id": "test_abort",
        "test_mode": True,
        "config_overrides": {"random_seed": profile.random_seed, "hardware_target": profile.hardware_target},
    }
    start = harness.request("abort", "start", "POST", "/research/start", start_payload, timeout_sec=timeout_sec)
    experiment_id = str(_extract_data(start["response"]["body"]).get("experiment_id", ""))
    if not experiment_id:
        raise RuntimeError("Abort scenario failed to start")
    answer_stats = {"llm_attempts": 0, "llm_success": 0, "fallback_used": 0}
    _answer_clarifications(
        harness=harness,
        scenario="abort",
        experiment_id=experiment_id,
        initial_pending=_extract_data(start["response"]["body"]).get("pending_questions") or {},
        profile=profile,
        rng=rng,
        run_tag=run_tag,
        answer_mode=answer_mode,
        llm_answer_timeout_sec=llm_answer_timeout_sec,
        answer_stats=answer_stats,
    )
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
    status_data = _extract_data(status["response"]["body"])
    detail_data = _extract_data(detail["response"]["body"])
    agents_seen = _collect_working_agents(
        _extract_data(logs["response"]["body"]),
        phase_timings=(detail_data.get("phase_timings") if isinstance(detail_data, dict) else None),
        final_phase=str(status_data.get("phase", "")),
    )
    return {
        "scenario": "abort",
        "experiment_id": experiment_id,
        "final_status": str(status_data.get("status", "")),
        "final_phase": str(status_data.get("phase", "")),
        "agents_seen": agents_seen,
        "llm_calls_count": int(detail_data.get("llm_calls_count", 0) or 0),
        "total_tokens_used": int(detail_data.get("total_tokens_used", 0) or 0),
        "answering_stats": answer_stats,
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


def _aggregate_answering_stats(scenario_results: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"llm_attempts": 0, "llm_success": 0, "fallback_used": 0}
    for item in scenario_results:
        stats = item.get("answering_stats")
        if isinstance(stats, dict):
            for key in totals:
                totals[key] += int(stats.get(key, 0) or 0)
    return totals


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
        help="Minimum acceptable avg LLM calls per experiment (scenario result first, then /system/metrics).",
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
        "--clarification-answer-mode",
        choices=["llm", "llm_strict", "heuristic"],
        default="llm",
        help="How clarification answers are generated: llm (fallback), llm_strict (no fallback), heuristic.",
    )
    parser.add_argument(
        "--llm-answer-timeout-sec",
        type=int,
        default=35,
        help="Timeout for each LLM clarification answer request.",
    )
    parser.add_argument(
        "--ensure-all-agents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, auto-include scenarios required to cover all agents.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic answers/profile (fixed seed 42 unless --seed is provided).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for dynamic clarification/profile generation.",
    )
    args = parser.parse_args()

    _enforce_local_base_url(args.base_url, allow_remote=args.allow_remote_base_url)
    seed = int(args.seed) if args.seed is not None else (42 if bool(args.deterministic) else int(time.time_ns() % 2_147_483_000))
    rng = random.Random(seed)
    run_tag = f"seed{seed}"
    selected = [part.strip().lower() for part in str(args.scenarios).split(",") if part.strip()]
    if args.ai_only:
        selected = [name for name in selected if name in {"standard", "recovery", "abort"}]
        if not selected:
            selected = ["standard", "recovery", "abort"]
    ensure_all_agents = bool(args.ensure_all_agents) and not bool(args.ai_only)
    if ensure_all_agents:
        for scenario_name in ("standard", "quantum", "recovery"):
            if scenario_name not in selected:
                selected.append(scenario_name)
    allowed = {"standard", "quantum", "recovery", "abort"}
    invalid = sorted([name for name in selected if name not in allowed])
    if invalid:
        print(f"ERROR: invalid scenarios: {invalid}", file=sys.stderr)
        return 2

    harness = ApiHarness(base_url=args.base_url)
    scenario_results: list[dict[str, Any]] = []
    scenario_errors: list[dict[str, str]] = []

    for scenario in selected:
        scenario_rng = random.Random(rng.randint(1, 2_147_483_000))
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
                        rng=scenario_rng,
                        run_tag=f"{run_tag}_{scenario}",
                        answer_mode=str(args.clarification_answer_mode),
                        llm_answer_timeout_sec=int(args.llm_answer_timeout_sec),
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
                        rng=scenario_rng,
                        run_tag=f"{run_tag}_{scenario}",
                        answer_mode=str(args.clarification_answer_mode),
                        llm_answer_timeout_sec=int(args.llm_answer_timeout_sec),
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
                        rng=scenario_rng,
                        run_tag=f"{run_tag}_{scenario}",
                        answer_mode=str(args.clarification_answer_mode),
                        llm_answer_timeout_sec=int(args.llm_answer_timeout_sec),
                    )
                )
            elif scenario == "abort":
                scenario_results.append(
                    _run_abort_scenario(
                        harness=harness,
                        timeout_sec=args.timeout_sec,
                        rng=scenario_rng,
                        run_tag=f"{run_tag}_{scenario}",
                        answer_mode=str(args.clarification_answer_mode),
                        llm_answer_timeout_sec=int(args.llm_answer_timeout_sec),
                    )
                )
        except Exception as exc:
            scenario_errors.append({"scenario": scenario, "error": str(exc)})

    expected_agents = _expected_agents_for_selected(selected_scenarios=selected, ensure_all_agents=ensure_all_agents)
    covered_agents = sorted({agent for item in scenario_results for agent in item.get("agents_seen", [])})
    missing_agents = sorted([agent for agent in expected_agents if agent not in covered_agents])
    latency_values = [float(record["response"]["elapsed_ms"]) for record in harness.records]
    endpoint_coverage = _endpoint_coverage(harness.records)
    rl_validation = _validate_rl_feedback(scenario_results)
    llm_call_depth = _validate_llm_call_depth(scenario_results, min_avg_calls=float(args.min_avg_llm_calls))
    clarification_answering = _aggregate_answering_stats(scenario_results)
    rl_gate_ok = (rl_validation.get("status") == "pass") if bool(args.require_rl_feedback) else True
    llm_gate_ok = llm_call_depth.get("status") == "pass"
    agent_gate_ok = not missing_agents

    output = {
        "generated_at": time.time(),
        "base_url": args.base_url,
        "selected_scenarios": selected,
        "mode": "ai_only" if args.ai_only else "mixed",
        "clarification_answer_mode": str(args.clarification_answer_mode),
        "llm_answer_timeout_sec": int(args.llm_answer_timeout_sec),
        "ensure_all_agents": ensure_all_agents,
        "answer_mode": "deterministic" if bool(args.deterministic) else "dynamic",
        "answer_seed": int(seed),
        "execute_install_actions": bool(args.execute_install_actions),
        "require_rl_feedback": bool(args.require_rl_feedback),
        "min_avg_llm_calls": float(args.min_avg_llm_calls),
        "requests_total": len(harness.records),
        "latency": _latency_stats(latency_values),
        "endpoint_coverage": endpoint_coverage,
        "expected_agents": expected_agents,
        "covered_agents": covered_agents,
        "missing_agents": missing_agents,
        "scenario_results": scenario_results,
        "scenario_errors": scenario_errors,
        "external_integrations": {
            "rl_feedback_validation": rl_validation,
            "llm_call_depth_validation": llm_call_depth,
            "clarification_answering": clarification_answering,
        },
        "requests": harness.records,
        "result": "success"
        if (not scenario_errors and rl_gate_ok and llm_gate_ok and agent_gate_ok)
        else "partial_failure",
    }

    output_path = Path(args.output_json).resolve()
    _save_json(output_path, output)
    print(f"Done. Single JSON report written to: {output_path}")
    return 0 if output["result"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
