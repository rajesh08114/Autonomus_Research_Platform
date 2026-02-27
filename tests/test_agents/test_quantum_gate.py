from __future__ import annotations

import pytest

from src.agents.quantum_gate import quantum_gate_node
from src.state.research_state import new_research_state


@pytest.mark.asyncio
async def test_quantum_gate_writes_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    async def _fake_invoke_master_llm(system_prompt: str, user_prompt: str = "", experiment_id: str | None = None, phase: str | None = None) -> str:
        _ = (system_prompt, user_prompt, experiment_id, phase)
        return (
            '{"code":"from __future__ import annotations\\n'
            'QUBIT_COUNT = 4\\nCIRCUIT_LAYERS = 3\\nBACKEND = \\"default.qubit\\"\\n\\n'
            'class QuantumLayer:\\n'
            '    def forward(self, x):\\n        return [0.0]\\n\\n'
            'def get_circuit_diagram():\\n    return \\"ok\\""}'
        )

    monkeypatch.setattr("src.agents.quantum_gate.invoke_master_llm", _fake_invoke_master_llm)

    state = new_research_state("exp_1", str(tmp_path), "quantum task", {})
    state["requires_quantum"] = True
    state["quantum_framework"] = "pennylane"
    state["data_report"] = {"shape": [100, 4], "class_distribution": {"0": 50, "1": 50}}
    state["research_plan"] = {}
    state = await quantum_gate_node(state)
    assert any(path.endswith("quantum_circuit.py") for path in state["created_files"])
