from __future__ import annotations

from pathlib import Path

from src.core.logger import get_logger
from src.state.research_state import ResearchState

logger = get_logger(__name__)

async def job_scheduler_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "job_scheduler"
    logger.info("agent.scheduler.start", experiment_id=state["experiment_id"])
    project = Path(state["project_path"])
    order = [
        str(project / "main.py"),
    ]
    # Keep only existing files in order.
    state["execution_order"] = [script for script in order if Path(script).exists()]
    logger.info("agent.scheduler.end", experiment_id=state["experiment_id"], execution_order=state["execution_order"])
    return state
