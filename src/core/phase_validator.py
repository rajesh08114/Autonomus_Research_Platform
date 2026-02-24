from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.config.settings import settings
from src.core.security import ensure_project_path
from src.state.research_state import ResearchState

PACKAGE_SPEC_RE = re.compile(r"^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-]+$")


@dataclass(slots=True)
class PhaseValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    phase: str


def _state_size_kb(state: ResearchState) -> float:
    raw = json.dumps(state, default=str).encode("utf-8")
    return len(raw) / 1024.0


def _ensure_all_paths_within_project(paths: list[str], project_path: str) -> list[str]:
    errors: list[str] = []
    for p in paths:
        try:
            ensure_project_path(p, project_path)
        except Exception:
            errors.append(f"Path outside project boundary: {p}")
    return errors


def _validate_common_security(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    size_kb = _state_size_kb(state)
    if size_kb > settings.MAX_STATE_SIZE_KB:
        errors.append(f"State size exceeds limit: {size_kb:.2f}KB > {settings.MAX_STATE_SIZE_KB}KB")

    errors.extend(_ensure_all_paths_within_project(state.get("created_files", []), state["project_path"]))
    if state.get("current_script"):
        errors.extend(_ensure_all_paths_within_project([state["current_script"]], state["project_path"]))

    for spec in state.get("required_packages", []):
        if not PACKAGE_SPEC_RE.match(spec):
            errors.append(f"Invalid package pin format: {spec}")

    if state.get("retry_count", 0) > settings.MAX_RETRY_COUNT:
        errors.append("Retry count exceeds configured cap")

    if state.get("requires_quantum") and not settings.QUANTUM_ENABLED:
        warnings.append("Quantum requested while QUANTUM_ENABLED=false")

    return errors, warnings


def _validate_clarifier(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    pending = state.get("pending_user_question") or {}
    current_question = pending.get("current_question")
    if not isinstance(current_question, dict):
        errors.append("Clarifier missing current_question")
        return errors, warnings

    questions = pending.get("questions", [])
    if not isinstance(questions, list) or len(questions) != 1:
        errors.append("Clarifier should expose exactly one active question in questions")

    asked = pending.get("asked_question_ids", [])
    if not isinstance(asked, list):
        errors.append("Clarifier asked_question_ids must be a list")
        asked = []

    total_planned = int(pending.get("total_questions_planned") or 0)
    if total_planned > 12:
        errors.append("Clarifier total_questions_planned exceeds hard cap (12)")

    for q in [current_question]:
        if not isinstance(q, dict):
            errors.append("Invalid question entry type")
            continue
        for required in ("id", "text", "type"):
            if required not in q:
                errors.append(f"Question missing field: {required}")
        if q.get("type") == "choice" and not q.get("options"):
            errors.append(f"Choice question missing options: {q.get('id')}")
    if current_question.get("id") in asked:
        warnings.append("Current question id already present in asked_question_ids")
    return errors, warnings


def _validate_planner(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    plan = state.get("research_plan", {})
    for key in ("objective", "methodology", "algorithm", "framework", "dataset", "metrics"):
        if key not in plan:
            errors.append(f"Planner missing research_plan.{key}")
    if not state.get("required_packages"):
        errors.append("Planner produced empty required_packages")
    if state.get("requires_quantum") and not state.get("quantum_framework"):
        warnings.append("Quantum required but framework not explicitly set")
    return errors, warnings


def _validate_env(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    waiting = str(state.get("status")) == "waiting_user"
    pending = state.get("pending_user_confirm")
    if waiting and not pending:
        errors.append("Env manager in waiting_user without pending_user_confirm")
    if pending:
        for field in ("action_id", "action", "package", "version"):
            if field not in pending:
                errors.append(f"Pending confirm missing {field}")
    return errors, warnings


def _validate_dataset(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    report = state.get("data_report", {})
    for key in ("shape", "columns"):
        if key not in report:
            errors.append(f"Dataset report missing {key}")
    dataset_path = state.get("dataset_path", "")
    if not dataset_path:
        errors.append("Dataset path not set")
    else:
        try:
            ensure_project_path(dataset_path, state["project_path"])
        except Exception:
            errors.append("Dataset path outside project")
    return errors, warnings


def _validate_codegen(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    project = Path(state["project_path"])
    required = [
        project / "config.py",
        project / "main.py",
        project / "src" / "utils.py",
        project / "src" / "preprocessing.py",
        project / "src" / "model.py",
        project / "src" / "train.py",
        project / "src" / "evaluate.py",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"Code generation missing file: {path.name}")
    return errors, warnings


def _validate_quantum(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if state.get("requires_quantum"):
        code = state.get("quantum_circuit_code")
        if not code:
            errors.append("Quantum gate did not persist quantum_circuit_code")
        elif "class QuantumLayer" not in code:
            errors.append("Quantum circuit code missing QuantumLayer")
        path = Path(state["project_path"]) / "src" / "quantum_circuit.py"
        if not path.exists():
            errors.append("Quantum gate did not create src/quantum_circuit.py")
    return errors, warnings


def _validate_scheduler(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    order = state.get("execution_order", [])
    if not order:
        errors.append("Scheduler produced empty execution_order")
    errors.extend(_ensure_all_paths_within_project(order, state["project_path"]))
    return errors, warnings


def _validate_runner(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    logs = state.get("execution_logs", [])
    if not logs:
        warnings.append("Subprocess runner has no execution logs yet")
        return errors, warnings
    latest = logs[-1]
    for key in ("script_path", "returncode", "stdout", "stderr", "duration_sec"):
        if key not in latest:
            errors.append(f"Execution log missing {key}")
    if len(latest.get("stdout", "")) > settings.STDOUT_CAP_CHARS:
        errors.append("stdout exceeds cap")
    if len(latest.get("stderr", "")) > settings.STDERR_CAP_CHARS:
        errors.append("stderr exceeds cap")
    return errors, warnings


def _validate_error_recovery(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if state.get("retry_count", 0) > settings.MAX_RETRY_COUNT:
        errors.append("Error recovery exceeded retry cap")
    if state.get("errors") and not state.get("last_error_category"):
        warnings.append("Errors exist but last_error_category not set")
    return errors, warnings


def _validate_evaluator(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not state.get("metrics"):
        errors.append("Evaluator did not populate metrics")
    summary = state.get("evaluation_summary", {})
    if "metrics" not in summary:
        warnings.append("Evaluation summary missing metrics section")
    return errors, warnings


def _validate_doc(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    doc_path = state.get("documentation_path")
    if not doc_path:
        errors.append("Doc generator did not set documentation_path")
        return errors, warnings
    try:
        ensure_project_path(doc_path, state["project_path"])
    except Exception:
        errors.append("documentation_path outside project")
    if not Path(doc_path).exists():
        errors.append("final report file missing")
    return errors, warnings


PHASE_VALIDATORS: dict[str, Callable[[ResearchState], tuple[list[str], list[str]]]] = {
    "clarifier": _validate_clarifier,
    "planner": _validate_planner,
    "env_manager": _validate_env,
    "dataset_manager": _validate_dataset,
    "code_generator": _validate_codegen,
    "quantum_gate": _validate_quantum,
    "job_scheduler": _validate_scheduler,
    "subprocess_runner": _validate_runner,
    "error_recovery": _validate_error_recovery,
    "results_evaluator": _validate_evaluator,
    "doc_generator": _validate_doc,
}


def validate_phase_output(phase: str, state: ResearchState) -> PhaseValidationResult:
    errors, warnings = _validate_common_security(state)
    phase_validator = PHASE_VALIDATORS.get(phase)
    if phase_validator:
        phase_errors, phase_warnings = phase_validator(state)
        errors.extend(phase_errors)
        warnings.extend(phase_warnings)
    return PhaseValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings, phase=phase)
