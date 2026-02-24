from __future__ import annotations

from typing import Literal

from src.config.settings import settings
from src.state.research_state import ResearchState


def route_quantum_or_direct(state: ResearchState) -> Literal["quantum", "no_quantum"]:
    return "quantum" if state["requires_quantum"] else "no_quantum"


def route_success_or_error(state: ResearchState) -> Literal["success", "error", "abort"]:
    last = state["execution_logs"][-1] if state["execution_logs"] else {}
    if last.get("returncode", -1) == 0:
        return "success"
    if state["retry_count"] >= settings.MAX_RETRY_COUNT:
        return "abort"
    return "error"


def route_retry_or_abort(state: ResearchState) -> Literal["retry", "abort"]:
    return "abort" if state["retry_count"] >= settings.MAX_RETRY_COUNT else "retry"

