from __future__ import annotations

from typing import Any

from src.core.execution_mode import normalize_execution_mode
from src.state.research_state import ResearchState

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enabled", "enable"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disabled", "disable"}


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in {0, 1}:
            return bool(value)
        return None
    text = str(value or "").strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_user_behavior_profile(state: ResearchState) -> dict[str, Any]:
    clarifications = state.get("clarifications") or {}
    plan = state.get("research_plan") or {}
    denied_actions = state.get("denied_actions") or []
    local_action_history = state.get("local_action_history") or []
    confirm_requested = _safe_int(state.get("confirmations_requested"), 0)
    confirm_processed = _safe_int(state.get("confirmations_processed"), 0)
    denied_count = len(denied_actions)
    denominator = max(confirm_processed, confirm_requested, 1)
    denial_rate = round(float(denied_count) / float(denominator), 3)
    confirm_count = sum(1 for item in local_action_history if str(item.get("decision", "")).strip().lower() == "confirm")
    gpu_name = str((state.get("local_hardware_profile") or {}).get("gpu_name") or "").strip()

    decision_style = "balanced"
    if denied_count >= 2 or denial_rate >= 0.35:
        decision_style = "cautious"
    elif confirm_count >= 3 and denied_count == 0:
        decision_style = "streamlined"

    auto_retry = _to_bool(clarifications.get("auto_retry_preference"))
    if auto_retry is None:
        auto_retry = True

    problem_type = str(plan.get("problem_type") or clarifications.get("problem_type") or "").strip().lower() or "classification"
    code_level = str(plan.get("code_level") or clarifications.get("code_level") or "").strip().lower() or "intermediate"

    adaptation_hints: list[str] = []
    if decision_style == "cautious":
        adaptation_hints.append("Prefer lower-risk steps and explicit rationale for approvals.")
    if str(state.get("hardware_target") or "cpu").strip().lower() == "cpu":
        adaptation_hints.append("Prefer CPU-safe defaults and avoid GPU-only assumptions.")
    if str(state.get("output_format") or ".py").strip().lower() in {".ipynb", "hybrid"}:
        adaptation_hints.append("Prefer notebook-friendly outputs and examples.")
    if code_level == "low":
        adaptation_hints.append("Prioritize readability and concise implementations.")
    if not adaptation_hints:
        adaptation_hints.append("Keep actions deterministic and aligned to current state constraints.")

    return {
        "profile_version": 1,
        "execution_mode": normalize_execution_mode(state.get("execution_mode")),
        "interaction": {
            "confirmations_requested": confirm_requested,
            "confirmations_processed": confirm_processed,
            "denied_actions": denied_count,
            "denial_rate": denial_rate,
            "local_actions_executed": len(local_action_history),
            "decision_style": decision_style,
        },
        "preferences": {
            "research_type": str(state.get("research_type") or "ai"),
            "problem_type": problem_type,
            "code_level": code_level,
            "dataset_source": str(state.get("dataset_source") or "sklearn"),
            "output_format": str(state.get("output_format") or ".py"),
            "hardware_target": str(state.get("hardware_target") or "cpu"),
            "auto_retry_enabled": bool(auto_retry),
        },
        "environment": {
            "local_python_command": str(state.get("local_python_command") or "python"),
            "gpu_available": bool(gpu_name),
            "gpu_name": gpu_name or None,
            "platform": str((state.get("local_hardware_profile") or {}).get("platform") or ""),
            "logical_cores": _safe_int((state.get("local_hardware_profile") or {}).get("logical_cores"), 0) or None,
            "memory_gb": (state.get("local_hardware_profile") or {}).get("total_memory_gb"),
        },
        "adaptation_hints": adaptation_hints,
    }
