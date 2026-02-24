from __future__ import annotations

import pytest

from src.agents.quantum_gate import quantum_gate_node
from src.state.research_state import new_research_state


@pytest.mark.asyncio
async def test_quantum_gate_writes_file(tmp_path):
    state = new_research_state("exp_1", str(tmp_path), "quantum task", {})
    state["requires_quantum"] = True
    state["quantum_framework"] = "pennylane"
    state["data_report"] = {"shape": [100, 4], "class_distribution": {"0": 50, "1": 50}}
    state["research_plan"] = {}
    state = await quantum_gate_node(state)
    assert any(path.endswith("quantum_circuit.py") for path in state["created_files"])

