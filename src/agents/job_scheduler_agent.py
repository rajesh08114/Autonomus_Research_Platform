from __future__ import annotations

from pathlib import Path

from src.core.logger import get_logger
from src.state.research_state import ResearchState

logger = get_logger(__name__)

async def job_scheduler_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "job_scheduler"
    logger.info("agent.scheduler.start", experiment_id=state["experiment_id"])
    project = Path(state["project_path"])
    code_level = str((state.get("research_plan") or {}).get("code_level", "intermediate")).strip().lower()
    algorithm_class = str((state.get("research_plan") or {}).get("algorithm_class", "supervised")).strip().lower()
    order: list[str] = []
    if code_level == "advanced":
        order.append(str(project / "data" / "validate_data.py"))
    if algorithm_class in {"reinforcement", "quantum_ml"}:
        order.append(str(project / "main.py"))
    else:
        order.extend([str(project / "main.py")])
    # Keep only existing files in order.
    state["execution_order"] = [script for script in order if Path(script).exists()]
    logger.info("agent.scheduler.end", experiment_id=state["experiment_id"], execution_order=state["execution_order"])
    return state
