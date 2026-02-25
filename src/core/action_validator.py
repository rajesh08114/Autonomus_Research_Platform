from __future__ import annotations

from typing import Any
import re

from src.state.research_state import ResearchState
from src.core.security import sanitize_subprocess_args

ALLOWED_ACTIONS = {
    "ask_user",
    "create_project",
    "create_file",
    "write_code",
    "modify_file",
    "install_package",
    "run_python",
    "analyze_results",
    "generate_documentation",
    "delegate_quantum_code",
    "finish",
    "abort",
}

REQUIRED_FIELDS = {
    "run_python": ["script_path", "timeout_seconds"],
    "delegate_quantum_code": ["framework", "algorithm", "qubit_count"],
    "write_code": ["file_path", "content"],
    "modify_file": ["file_path", "find", "replace"],
    "install_package": ["package", "version"],
    "ask_user": ["questions"],
}

PATH_RESTRICTED_ACTIONS = {"write_code", "modify_file", "run_python", "create_file"}
PINNED_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
DANGEROUS_CODE_MARKERS = [
    "shell=True",
    "os.system(",
    "subprocess.Popen(",
    "eval(",
    "exec(",
]

PHASE_ALLOWED_ACTIONS = {
    "clarifier": {"ask_user"},
    "planner": {"create_project"},
    "env_manager": {"install_package", "ask_user"},
    "dataset_manager": {"write_code", "run_python", "ask_user"},
    "code_generator": {"write_code", "delegate_quantum_code"},
    "quantum_gate": {"write_code", "finish"},
    "job_scheduler": {"run_python"},
    "error_recovery": {"modify_file", "install_package", "ask_user", "abort"},
    "results_evaluator": {"analyze_results", "run_python"},
    "doc_generator": {"generate_documentation", "finish"},
}


def _validate_phase_action(action_name: str, phase: str | None) -> tuple[bool, str]:
    if not phase:
        return True, "ok"
    allowed = PHASE_ALLOWED_ACTIONS.get(phase)
    if not allowed:
        return True, "ok"
    if action_name not in allowed:
        return False, f"Action {action_name} not allowed in phase {phase}"
    return True, "ok"


def validate_action(action: dict[str, Any], state: ResearchState, phase: str | None = None) -> tuple[bool, str]:
    action_name = action.get("action")
    if action_name not in ALLOWED_ACTIONS:
        return False, f"Unknown action: {action_name}"

    phase_ok, phase_err = _validate_phase_action(action_name, phase)
    if not phase_ok:
        return False, phase_err

    reasoning = action.get("reasoning", "")
    if not reasoning or len(reasoning) < 10:
        return False, "reasoning field missing or too short"

    if not action.get("next_step"):
        return False, "next_step is required"

    confidence = action.get("confidence")
    if confidence is not None:
        try:
            value = float(confidence)
        except Exception:
            return False, "confidence must be numeric"
        if value < 0.0 or value > 1.0:
            return False, "confidence must be in [0.0, 1.0]"

    params = action.get("parameters", {})
    if not isinstance(params, dict):
        return False, "parameters must be an object"

    for required in REQUIRED_FIELDS.get(action_name, []):
        if required not in params:
            return False, f"Missing required parameter: {required}"

    if action_name in PATH_RESTRICTED_ACTIONS:
        path = params.get("file_path") or params.get("script_path")
        if path and not str(path).startswith(state["project_path"]):
            return False, f"Path traversal blocked: {path}"

    if action_name == "write_code":
        content = str(params.get("content", ""))
        for marker in DANGEROUS_CODE_MARKERS:
            if marker in content:
                return False, f"Unsafe code marker detected in write_code content: {marker}"

    if action_name == "run_python":
        args = params.get("args", [])
        if args and not isinstance(args, list):
            return False, "run_python.args must be an array"
        try:
            sanitize_subprocess_args([str(a) for a in args])
        except Exception as exc:
            return False, f"Unsafe run_python args: {exc}"

    if action_name == "install_package":
        package = str(params.get("package", ""))
        version = str(params.get("version", ""))
        if package != "__complete__" and not PINNED_PACKAGE_RE.match(package):
            return False, f"Unsafe package name: {package}"
        if package != "__complete__" and (not version or not PINNED_PACKAGE_RE.match(version)):
            return False, f"Unsafe package version: {version}"

    if action_name == "ask_user":
        questions = params.get("questions")
        if not isinstance(questions, list):
            return False, "ask_user.questions must be an array"
        for idx, item in enumerate(questions):
            if not isinstance(item, dict):
                return False, f"ask_user.questions[{idx}] must be an object"
            for field in ("id", "text", "type"):
                if field not in item:
                    return False, f"ask_user.questions[{idx}] missing {field}"

    return True, "ok"
