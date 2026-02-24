from __future__ import annotations

from copy import deepcopy

from src.state.research_state import ResearchState


def compress_state(state: ResearchState) -> dict:
    compressed = deepcopy(state)
    if len(compressed.get("execution_logs", [])) > 3:
        compressed["execution_logs"] = compressed["execution_logs"][-3:]
    if len(compressed.get("errors", [])) > 3:
        compressed["errors"] = compressed["errors"][-3:]
    if len(compressed.get("repair_history", [])) > 3:
        compressed["repair_history"] = compressed["repair_history"][-3:]
    return compressed

