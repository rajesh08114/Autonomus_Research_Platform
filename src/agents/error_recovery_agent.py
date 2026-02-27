from __future__ import annotations

import time
from pathlib import Path

from src.config.settings import settings
from src.core.file_manager import replace_in_file
from src.core.logger import get_logger
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

async def error_recovery_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "error_recovery"
    logger.warning("agent.error_recovery.start", experiment_id=state["experiment_id"], retry_count=state["retry_count"])
    if state["retry_count"] >= settings.MAX_RETRY_COUNT:
        state["status"] = ExperimentStatus.ABORTED.value
        state["timestamp_end"] = time.time()
        return state

    if not state["errors"]:
        return state

    latest = state["errors"][-1]
    category = latest["category"]
    logger.warning("agent.error_recovery.classified", experiment_id=state["experiment_id"], category=category)
    state["retry_count"] += 1
    state["last_error_category"] = category

    if category in {"ModuleNotFoundError", "ImportError"}:
        # Let env manager resolve missing dependency on next pass.
        missing = "scikit-learn==1.8.0"
        if missing not in state["required_packages"]:
            state["required_packages"].append(missing)
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": "Added required package to required_packages",
                "file_changed": "requirements",
                "find_text": "",
                "replace_text": missing,
                "timestamp": time.time(),
            }
        )
        state["phase"] = "env_manager"
        logger.info("agent.error_recovery.fix_added_package", experiment_id=state["experiment_id"], package=missing)
        return state

    if category in {"gpu_unavailable", "RuntimeError"}:
        config_path = str(Path(state["project_path"]) / "config.py")
        ok = replace_in_file(state["project_path"], config_path, 'DEVICE = "cpu" if "', 'DEVICE = "cpu" if "', backup=True)
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": "Forced CPU execution",
                "file_changed": config_path,
                "find_text": "DEVICE",
                "replace_text": "cpu",
                "timestamp": time.time(),
            }
        )
        if not ok:
            state["status"] = ExperimentStatus.ABORTED.value
            state["timestamp_end"] = time.time()
            return state
        logger.info("agent.error_recovery.fix_cpu", experiment_id=state["experiment_id"], file=config_path)

    if category in {"SyntaxError", "IndentationError"}:
        plan = state.setdefault("research_plan", {})
        old_level = str(plan.get("code_level", "intermediate")).strip().lower()
        next_level = "intermediate" if old_level == "advanced" else "low"
        plan["code_level"] = next_level
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": f"Downgraded code generation complexity from {old_level} to {next_level}",
                "file_changed": "research_plan",
                "find_text": "code_level",
                "replace_text": next_level,
                "timestamp": time.time(),
            }
        )
        state["phase"] = "code_generator"
        logger.info("agent.error_recovery.fix_codegen_level", experiment_id=state["experiment_id"], from_level=old_level, to_level=next_level)
        return state

    if category in {"quantum_backend_error", "DeviceError"} and state.get("requires_quantum"):
        state["quantum_backend"] = "default.qubit"
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": "Switched quantum backend to default.qubit",
                "file_changed": "state.quantum_backend",
                "find_text": str(state.get("quantum_backend")),
                "replace_text": "default.qubit",
                "timestamp": time.time(),
            }
        )
        state["phase"] = "quantum_gate"
        logger.info("agent.error_recovery.fix_quantum_backend", experiment_id=state["experiment_id"], backend="default.qubit")
        return state

    if state["retry_count"] >= settings.MAX_RETRY_COUNT:
        state["status"] = ExperimentStatus.ABORTED.value
        state["timestamp_end"] = time.time()
        logger.error("agent.error_recovery.abort", experiment_id=state["experiment_id"], retry_count=state["retry_count"])
    return state
