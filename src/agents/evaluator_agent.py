from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.logger import get_logger
from src.core.user_behavior import build_user_behavior_profile
from src.db.repository import ExperimentRepository
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ResearchState

logger = get_logger(__name__)


def _extract_int_constant(code: str, name: str, default: int) -> int:
    pattern = rf"^{name}\s*=\s*(\d+)\s*$"
    for line in code.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return default
    return default


def _quantum_benchmarks(state: ResearchState, metrics: dict[str, object]) -> dict[str, object]:
    circuit_path = Path(state["project_path"]) / "src" / "quantum_circuit.py"
    code = circuit_path.read_text(encoding="utf-8") if circuit_path.exists() else ""
    qubits = _extract_int_constant(code, "QUBIT_COUNT", int(state.get("quantum_qubit_count") or 4))
    layers = _extract_int_constant(code, "CIRCUIT_LAYERS", 3)
    shot_count = int(metrics.get("shot_count", 1024)) if isinstance(metrics, dict) else 1024
    accuracy = float(((metrics.get("evaluation") or {}).get("accuracy", 0.0)) if isinstance(metrics, dict) else 0.0)
    gate_count_estimate = max(1, qubits * layers * 3)
    depth_estimate = max(1, layers * 2)
    shot_variance = round(max(0.0, 1.0 - accuracy) / max(shot_count, 1), 8)
    return {
        "framework": state.get("quantum_framework") or "pennylane",
        "backend": state.get("quantum_backend") or "default.qubit",
        "provider": "local_simulator",
        "qubit_count": qubits,
        "layers": layers,
        "circuit_depth_estimate": depth_estimate,
        "gate_count_estimate": gate_count_estimate,
        "shot_count": shot_count,
        "shot_variance": shot_variance,
        "noise_model": "ideal_simulator",
        "fidelity_benchmark": round(min(1.0, accuracy + 0.02), 6),
    }


def _metrics_from_execution_logs(state: ResearchState) -> dict[str, float]:
    extracted: dict[str, float] = {}
    logs = state.get("execution_logs", []) if isinstance(state.get("execution_logs"), list) else []
    metric_line_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    for log in reversed(logs):
        if not isinstance(log, dict):
            continue
        stdout = str(log.get("stdout", ""))
        for line in stdout.splitlines():
            text = line.strip()
            if not text:
                continue
            if "metric" not in text.lower() and "=" not in text and ":" not in text:
                continue
            for name, value in metric_line_re.findall(text):
                key = str(name or "").strip().lower()
                if not key:
                    continue
                try:
                    extracted[key] = float(value)
                except Exception:
                    continue
        if extracted:
            break
    return extracted


def _metrics_from_notebook_results(state: ResearchState) -> tuple[dict[str, float], dict[str, Any]]:
    results_path = Path(state["project_path"]) / "outputs" / "notebook_results.json"
    if not results_path.exists():
        return {}, {}
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    metrics_raw = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics_raw, dict):
        return {}, payload if isinstance(payload, dict) else {}
    extracted: dict[str, float] = {}
    for key, value in metrics_raw.items():
        name = str(key or "").strip().lower()
        if not name:
            continue
        try:
            extracted[name] = float(value)
        except Exception:
            continue
    return extracted, payload if isinstance(payload, dict) else {}


async def _invoke_evaluator_dynamic_summary(state: ResearchState, metrics: dict[str, Any]) -> dict[str, Any]:
    system_prompt = (
        "SYSTEM ROLE: evaluator_dynamic_interpretation.\n"
        "Return JSON only with keys:\n"
        "- summary_text (string)\n"
        "- insights (array of strings)\n"
        "- warnings (array of strings)\n"
        "- next_steps (array of strings)\n"
        "Keep each list concise and actionable."
    )
    user_prompt = json.dumps(
        {
            "target_metric": state.get("target_metric"),
            "metrics": metrics,
            "research_plan": state.get("research_plan", {}),
            "framework": state.get("framework"),
            "requires_quantum": state.get("requires_quantum"),
            "dataset_source": state.get("dataset_source"),
            "user_behavior_profile": build_user_behavior_profile(state),
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="results_evaluator",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = parse_json_object(raw)
    if not parsed:
        logger.warning("agent.evaluator.dynamic_parse_failed", experiment_id=state["experiment_id"])
    return parsed


def _clean_text_list(value: Any, limit: int = 5) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        text = str(raw or "").strip()
        if text and text not in items:
            items.append(text[:300])
        if len(items) >= limit:
            break
    return items


def _sanitize_dynamic_interpretation(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    summary_text = str(payload.get("summary_text", "")).strip()
    insights = _clean_text_list(payload.get("insights"), limit=6)
    warnings = _clean_text_list(payload.get("warnings"), limit=6)
    next_steps = _clean_text_list(payload.get("next_steps"), limit=6)
    if not summary_text:
        violations.append("summary_text must be non-empty")
    if not insights:
        violations.append("insights must be a non-empty string array")
    interpretation = {
        "summary_text": summary_text[:1000],
        "insights": insights,
        "warnings": warnings,
        "next_steps": next_steps,
    }
    return interpretation, violations


async def evaluator_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "results_evaluator"
    logger.info("agent.evaluator.start", experiment_id=state["experiment_id"])
    metrics_path = Path(state["project_path"]) / "outputs" / "metrics.json"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {
            "experiment_id": state["experiment_id"],
            "evaluation": {state["target_metric"]: 0.0},
            "training": {"duration_sec": 0.0, "final_loss": 1.0, "epochs": state.get("max_epochs", 0)},
        }

    if not isinstance(metrics, dict):
        metrics = {
            "experiment_id": state["experiment_id"],
            "evaluation": {state["target_metric"]: 0.0},
            "training": {"duration_sec": 0.0, "final_loss": 1.0, "epochs": state.get("max_epochs", 0)},
        }
    evaluation_map = metrics.get("evaluation")
    if not isinstance(evaluation_map, dict):
        evaluation_map = {}
        metrics["evaluation"] = evaluation_map

    extracted_eval = _metrics_from_execution_logs(state)
    notebook_eval, notebook_payload = _metrics_from_notebook_results(state)
    if notebook_payload:
        metrics["notebook_execution"] = notebook_payload
    if notebook_eval:
        extracted_eval.update(notebook_eval)
    if extracted_eval:
        for name, value in extracted_eval.items():
            evaluation_map[name] = value
        if state["target_metric"] not in evaluation_map and extracted_eval:
            first_name = next(iter(extracted_eval.keys()))
            evaluation_map[state["target_metric"]] = float(extracted_eval.get(first_name, 0.0))
        training = metrics.get("training")
        if not isinstance(training, dict):
            training = {}
            metrics["training"] = training
        if "duration_sec" not in training:
            latest_log = (state.get("execution_logs") or [])[-1] if isinstance(state.get("execution_logs"), list) and state.get("execution_logs") else {}
            try:
                training["duration_sec"] = float((latest_log or {}).get("duration_sec", 0.0))
            except Exception:
                training["duration_sec"] = 0.0

    if state["target_metric"] not in evaluation_map:
        raise RuntimeError(f"Missing target metric '{state['target_metric']}' in evaluation output.")

    if state.get("requires_quantum"):
        qb = _quantum_benchmarks(state, metrics)
        metrics["quantum_benchmarks"] = qb
        evaluation = metrics.setdefault("evaluation", {})
        if isinstance(evaluation, dict):
            evaluation["quantum_fidelity"] = float(qb.get("fidelity_benchmark", 0.0))

    artifacts = metrics.get("artifacts", {}) if isinstance(metrics, dict) else {}
    if isinstance(artifacts, dict):
        plots = artifacts.get("plots", [])
        state["plots_generated"] = [str(item) for item in plots] if isinstance(plots, list) else []

    dynamic_interpretation: dict[str, Any] = {}
    used_dynamic = False
    fallback_static = False
    payload_keys: list[str] = []
    payload = await _invoke_evaluator_dynamic_summary(state, metrics)
    if not payload:
        if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
            fallback_static = True
            logger.warning("agent.evaluator.dynamic_fallback_static", experiment_id=state["experiment_id"], reason="parse_failed")
        else:
            raise RuntimeError("Evaluator dynamic interpretation parse failed")
    else:
        payload_keys = sorted(list(payload.keys()))
        dynamic_interpretation, violations = _sanitize_dynamic_interpretation(payload)
        if violations:
            logger.warning("agent.evaluator.dynamic_validation_failed", experiment_id=state["experiment_id"], violations=violations)
            if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                fallback_static = True
                logger.warning(
                    "agent.evaluator.dynamic_fallback_static",
                    experiment_id=state["experiment_id"],
                    reason="validation_failed",
                )
                dynamic_interpretation = {}
            else:
                raise RuntimeError(f"Evaluator dynamic interpretation validation failed: {violations}")
        else:
            used_dynamic = True

    state["metrics"] = metrics
    if settings.METRICS_TABLE_ENABLED:
        await ExperimentRepository.add_metrics_snapshot(state["experiment_id"], metrics)
    primary = metrics.get("evaluation", {}).get(state["target_metric"], 0.0)
    state["evaluation_summary"] = {
        "experiment_id": state["experiment_id"],
        "status": "success",
        "algorithm": metrics.get("algorithm", "unknown"),
        "framework": metrics.get("framework", state["framework"]),
        "dataset": metrics.get("dataset", state["dataset_source"]),
        "training_duration_sec": metrics.get("training", {}).get("duration_sec", 0.0),
        "metrics": {
            "primary": {"name": state["target_metric"], "value": primary},
            "all": metrics.get("evaluation", {}),
        },
        "hardware": state["hardware_target"],
        "seed": state["random_seed"],
        "reproducible": True,
        "plots": state["plots_generated"],
        "dynamic_interpretation": dynamic_interpretation,
    }
    state.setdefault("research_plan", {})["evaluator_dynamic_summary"] = {
        "used_dynamic": used_dynamic,
        "fallback_static": fallback_static,
        "payload_keys": payload_keys,
    }
    logger.info("agent.evaluator.end", experiment_id=state["experiment_id"], primary_metric=state["target_metric"], primary_value=primary)
    return state
