from __future__ import annotations

from pathlib import Path

from src.core.file_manager import write_text_file
from src.core.logger import get_logger
from src.llm.quantum_llm import generate_quantum_code
from src.state.research_state import ResearchState

logger = get_logger(__name__)

async def quantum_gate_node(state: ResearchState) -> ResearchState:
    state["phase"] = "quantum_gate"
    if not state["requires_quantum"]:
        return state
    logger.info("agent.quantum_gate.start", experiment_id=state["experiment_id"], framework=state.get("quantum_framework"))

    delegation_spec = {
        "framework": state.get("quantum_framework") or "pennylane",
        "algorithm": state.get("quantum_algorithm") or "VQE",
        "qubit_count": state.get("quantum_qubit_count") or 4,
        "layers": state["research_plan"].get("circuit_layers", 3),
        "dataset_info": {
            "n_features": max(int(state["data_report"].get("shape", [0, 4])[1]) - 1, 1),
            "n_classes": len(state["data_report"].get("class_distribution", {"0": 0, "1": 0})),
            "encoding": state["research_plan"].get("encoding", "angle_encoding"),
        },
        "training_strategy": state["research_plan"].get("training_strategy", "hybrid"),
        "optimizer": state["research_plan"].get("optimizer", "adam"),
        "backend": state.get("quantum_backend") or "default.qubit",
        "return_expectation": "PauliZ",
        "integration_point": "model.py::QuantumLayer.forward()",
    }

    code = await generate_quantum_code(delegation_spec, experiment_id=state["experiment_id"])
    if "class QuantumLayer" not in code:
        raise ValueError("Quantum LLM response missing QuantumLayer class")

    target = Path(state["project_path"]) / "src" / "quantum_circuit.py"
    write_text_file(state["project_path"], str(target), code)
    state["quantum_circuit_code"] = code
    if str(target) not in state["created_files"]:
        state["created_files"].append(str(target))
    state["llm_calls_count"] += 1
    logger.info("agent.quantum_gate.end", experiment_id=state["experiment_id"], file_path=str(target), code_len=len(code))
    return state
