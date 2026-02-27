from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode
from src.core.file_manager import write_text_file
from src.core.local_actions import queue_local_file_action
from src.core.logger import get_logger
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ResearchState

logger = get_logger(__name__)

_FRAMEWORK_IMPORT_ROOTS: dict[str, set[str]] = {
    "pennylane": {"pennylane"},
    "qiskit": {"qiskit"},
    "cirq": {"cirq"},
}


def _extract_generated_code(payload: dict[str, Any]) -> str:
    for key in ("code", "generated_code", "content", "response"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        for key in ("code", "generated_code", "content", "response"):
            value = parameters.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _normalize_import_root(spec: str) -> str:
    raw = str(spec or "").strip().lower()
    if not raw:
        return ""
    raw = raw.split("==", 1)[0]
    raw = raw.split("[", 1)[0]
    raw = raw.replace("-", "_")
    return raw.split(".", 1)[0]


def _allowed_import_roots(state: ResearchState) -> set[str]:
    allowed = set(getattr(sys, "stdlib_module_names", set()))
    allowed.update({"src", "config"})
    for spec in state.get("required_packages") or []:
        root = _normalize_import_root(str(spec))
        if root:
            allowed.add(root)
    qf = str(state.get("quantum_framework") or "pennylane").strip().lower()
    allowed.update(_FRAMEWORK_IMPORT_ROOTS.get(qf, set()))
    return allowed


def _extract_constant_int(code: str, name: str) -> int | None:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", re.MULTILINE)
    match = pattern.search(code)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _extract_constant_text(code: str, name: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*['\"]([^'\"]+)['\"]\s*$", re.MULTILINE)
    match = pattern.search(code)
    return str(match.group(1)).strip() if match else ""


def _validate_quantum_code(state: ResearchState, code: str) -> list[str]:
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]

    class_found = False
    forward_found = False
    diagram_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QuantumLayer":
            class_found = True
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    forward_found = True
        if isinstance(node, ast.FunctionDef) and node.name == "get_circuit_diagram":
            diagram_found = True
    if not class_found:
        violations.append("missing class QuantumLayer")
    if not forward_found:
        violations.append("missing QuantumLayer.forward")
    if not diagram_found:
        violations.append("missing get_circuit_diagram")

    qcount = _extract_constant_int(code, "QUBIT_COUNT")
    layers = _extract_constant_int(code, "CIRCUIT_LAYERS")
    backend = _extract_constant_text(code, "BACKEND")
    if qcount is None:
        violations.append("missing QUBIT_COUNT integer constant")
    if layers is None:
        violations.append("missing CIRCUIT_LAYERS integer constant")
    if not backend:
        violations.append("missing BACKEND string constant")

    expected_qubits = int(state.get("quantum_qubit_count") or 4)
    if qcount is not None and qcount != expected_qubits:
        violations.append(f"QUBIT_COUNT mismatch: expected {expected_qubits}, got {qcount}")

    allowed_roots = _allowed_import_roots(state)
    disallowed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = str(alias.name or "").split(".", 1)[0].strip()
                if root and root not in allowed_roots:
                    disallowed.add(root)
        elif isinstance(node, ast.ImportFrom):
            if int(getattr(node, "level", 0) or 0) > 0:
                continue
            module = str(getattr(node, "module", "") or "").strip()
            root = module.split(".", 1)[0] if module else ""
            if root and root not in allowed_roots:
                disallowed.add(root)
    if disallowed:
        violations.append(f"disallowed import roots: {sorted(disallowed)}")
    return violations


async def _invoke_quantum_llm(
    *,
    state: ResearchState,
    delegation_spec: dict[str, Any],
    repair_violations: list[str] | None = None,
    existing_code: str = "",
) -> dict[str, Any]:
    repair_violations = list(repair_violations or [])
    system_prompt = (
        "SYSTEM ROLE: quantum_gate_dynamic_code.\n"
        "Return one JSON object only with keys:\n"
        "- code: python module string\n"
        "- reasoning: short string\n"
        "Code requirements:\n"
        "- define class QuantumLayer with method forward(self, x)\n"
        "- define get_circuit_diagram()\n"
        "- include constants: QUBIT_COUNT (int), CIRCUIT_LAYERS (int), BACKEND (str)\n"
        "- use only imports allowed by state context.\n"
    )
    if repair_violations:
        system_prompt += "This is a repair turn. Fix every listed violation exactly.\n"
    user_prompt = json.dumps(
        {
            "delegation_spec": delegation_spec,
            "allowed_import_roots": sorted(_allowed_import_roots(state)),
            "repair_violations": repair_violations,
            "existing_code": existing_code,
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="quantum_gate",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    payload = parse_json_object(raw)
    if not payload:
        logger.warning("agent.quantum_gate.dynamic_parse_failed", experiment_id=state["experiment_id"])
    return payload


async def _generate_quantum_code_strict(state: ResearchState, delegation_spec: dict[str, Any]) -> tuple[str, list[str], int]:
    payload = await _invoke_quantum_llm(state=state, delegation_spec=delegation_spec)
    code = _extract_generated_code(payload)
    violations = _validate_quantum_code(state, code) if code else ["missing quantum code content"]
    if violations:
        logger.warning("agent.quantum_gate.dynamic_validation_failed", experiment_id=state["experiment_id"], violations=violations)

    repair_attempts = max(0, int(settings.STRICT_STATE_ONLY_REPAIR_ATTEMPTS))
    attempts_used = 0
    while violations and attempts_used < repair_attempts:
        attempts_used += 1
        payload = await _invoke_quantum_llm(
            state=state,
            delegation_spec=delegation_spec,
            repair_violations=violations,
            existing_code=code,
        )
        new_code = _extract_generated_code(payload)
        if new_code:
            code = new_code
        violations = _validate_quantum_code(state, code) if code else ["missing quantum code content"]
        if violations:
            logger.warning(
                "agent.quantum_gate.dynamic_validation_failed",
                experiment_id=state["experiment_id"],
                attempt=attempts_used,
                violations=violations,
            )
    return code, violations, attempts_used


async def quantum_gate_node(state: ResearchState) -> ResearchState:
    state["phase"] = "quantum_gate"
    if not state["requires_quantum"]:
        return state
    logger.info("agent.quantum_gate.start", experiment_id=state["experiment_id"], framework=state.get("quantum_framework"))
    problem_type = str((state.get("research_plan") or {}).get("problem_type", "classification"))
    code_level = str((state.get("research_plan") or {}).get("code_level", "intermediate"))
    layer_hint = 2 if code_level == "low" else (5 if code_level == "advanced" else 3)

    delegation_spec = {
        "framework": state.get("quantum_framework") or "pennylane",
        "algorithm": state.get("quantum_algorithm") or "VQE",
        "qubit_count": state.get("quantum_qubit_count") or 4,
        "layers": state["research_plan"].get("circuit_layers", layer_hint),
        "dataset_info": {
            "n_features": max(int(state["data_report"].get("shape", [0, 4])[1]) - 1, 1),
            "n_classes": len(state["data_report"].get("class_distribution", {"0": 0, "1": 0})),
            "encoding": state["research_plan"].get("encoding", "angle_encoding"),
        },
        "problem_type": problem_type,
        "code_level": code_level,
        "training_strategy": state["research_plan"].get("training_strategy", "hybrid"),
        "optimizer": state["research_plan"].get("optimizer", "adam"),
        "backend": state.get("quantum_backend") or "default.qubit",
        "return_expectation": "PauliZ",
        "integration_point": "model.py::QuantumLayer.forward()",
    }

    code, violations, repair_attempts_used = await _generate_quantum_code_strict(state, delegation_spec)
    state.setdefault("research_plan", {})["quantum_dynamic_summary"] = {
        "repair_attempts_used": repair_attempts_used,
        "strict_violations": violations,
        "framework": state.get("quantum_framework"),
        "algorithm": state.get("quantum_algorithm"),
    }
    if violations:
        raise RuntimeError(f"Quantum dynamic generation failed strict validation: {violations}")

    state["quantum_circuit_code"] = code
    target = Path(state["project_path"]) / "src" / "quantum_circuit.py"
    if is_vscode_execution_mode(state):
        planned = {"path": str(target), "content": code, "phase": "quantum_gate"}
        state.setdefault("local_file_plan", []).append(planned)
        if str(target) not in state["created_files"]:
            state["created_files"].append(str(target))
        queued = queue_local_file_action(
            state=state,
            phase="quantum_gate",
            file_operations=[planned],
            next_phase="job_scheduler",
            reason="Create quantum circuit source locally before job scheduling",
            cwd=state["project_path"],
        )
        if queued:
            logger.info(
                "agent.quantum_gate.pending_local_action",
                experiment_id=state["experiment_id"],
                file_path=str(target),
            )
            return state
    else:
        write_text_file(state["project_path"], str(target), code)
    if not is_vscode_execution_mode(state) and str(target) not in state["created_files"]:
        state["created_files"].append(str(target))
    logger.info("agent.quantum_gate.end", experiment_id=state["experiment_id"], file_path=str(target), code_len=len(code))
    return state

