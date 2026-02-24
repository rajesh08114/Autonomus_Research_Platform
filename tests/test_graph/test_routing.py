from __future__ import annotations

from src.graph.routers import route_quantum_or_direct, route_retry_or_abort, route_success_or_error
from src.state.research_state import new_research_state


def test_route_quantum_or_direct(tmp_path):
    state = new_research_state("exp_1", str(tmp_path), "x", {})
    state["requires_quantum"] = True
    assert route_quantum_or_direct(state) == "quantum"


def test_route_success_or_error(tmp_path):
    state = new_research_state("exp_1", str(tmp_path), "x", {})
    state["execution_logs"] = [{"script_path": "a", "returncode": 0, "stdout": "", "stderr": "", "duration_sec": 0.1, "timestamp": 0.0}]
    assert route_success_or_error(state) == "success"


def test_route_retry_or_abort(tmp_path):
    state = new_research_state("exp_1", str(tmp_path), "x", {})
    state["retry_count"] = 0
    assert route_retry_or_abort(state) == "retry"

