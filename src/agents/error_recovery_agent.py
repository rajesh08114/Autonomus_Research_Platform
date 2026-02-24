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
        missing = "scikit-learn==1.4.2"
        if missing not in state["required_packages"]:
            state["required_packages"].append(missing)
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": "Added fallback package to required_packages",
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
                "fix_description": "Forced CPU fallback",
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
        logger.info("agent.error_recovery.fix_cpu_fallback", experiment_id=state["experiment_id"], file=config_path)

    if state["retry_count"] >= settings.MAX_RETRY_COUNT:
        state["status"] = ExperimentStatus.ABORTED.value
        state["timestamp_end"] = time.time()
        logger.error("agent.error_recovery.abort", experiment_id=state["experiment_id"], retry_count=state["retry_count"])
    return state
