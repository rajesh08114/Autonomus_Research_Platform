from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.config.settings import settings
from src.core.execution_mode import BACKEND_MODE, VSCODE_EXTENSION_MODE, is_vscode_execution_mode
from src.core.security import ensure_project_path
from src.state.research_state import ResearchState

PACKAGE_SPEC_RE = re.compile(r"^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-]+$")
ALLOWED_QUESTION_TYPES = {"choice", "text", "boolean", "number"}
ALLOWED_PROBLEM_TYPES = {"classification", "regression", "clustering", "reinforcement", "forecasting", "generation"}
ALLOWED_CODE_LEVELS = {"low", "intermediate", "advanced"}


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


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


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
    execution_mode = str(state.get("execution_mode", "")).strip().lower()
    if execution_mode not in {VSCODE_EXTENSION_MODE, BACKEND_MODE}:
        warnings.append(f"Unknown execution_mode '{execution_mode}', default behavior may apply")

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
    if total_planned <= 0:
        errors.append("Clarifier total_questions_planned must be > 0")

    for q in [current_question]:
        if not isinstance(q, dict):
            errors.append("Invalid question entry type")
            continue
        for required in ("id", "text", "type"):
            if required not in q:
                errors.append(f"Question missing field: {required}")
        qtype = str(q.get("type", "")).strip().lower()
        if qtype not in ALLOWED_QUESTION_TYPES:
            errors.append(f"Question has unsupported type: {q.get('type')}")
        topic = str(q.get("topic", "")).strip()
        if not topic:
            errors.append("Question missing topic")
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
    clar = state.get("clarifications", {})
    problem_type = str(plan.get("problem_type") or clar.get("problem_type") or "").strip().lower()
    if not problem_type:
        errors.append("Planner missing problem_type in research plan")
    elif problem_type not in ALLOWED_PROBLEM_TYPES:
        errors.append(f"Planner problem_type unsupported: {problem_type}")
    code_level = str(plan.get("code_level") or clar.get("code_level") or "").strip().lower()
    if not code_level:
        errors.append("Planner missing code_level in research plan")
    elif code_level not in ALLOWED_CODE_LEVELS:
        errors.append(f"Planner code_level unsupported: {code_level}")
    if state.get("dataset_source") == "kaggle" and not state.get("kaggle_dataset_id"):
        warnings.append("Dataset source is kaggle but kaggle_dataset_id is empty")
    if state.get("dataset_source") == "upload":
        report = state.get("data_report", {})
        source = report.get("source", {}) if isinstance(report, dict) else {}
        if isinstance(source, dict) and source.get("resolved_source") == "synthetic":
            warnings.append("Upload source fell back to synthetic dataset")
    return errors, warnings


def _validate_env(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    waiting = str(state.get("status")) == "waiting_user"
    pending = state.get("pending_user_confirm")
    if waiting and not pending:
        errors.append("Env manager in waiting_user without pending_user_confirm")
    if pending:
        for field in ("action_id", "action"):
            if field not in pending:
                errors.append(f"Pending confirm missing {field}")
        if "cwd" in pending:
            errors.extend(_ensure_all_paths_within_project([str(pending.get("cwd", ""))], state["project_path"]))
        if pending.get("action") == "install_package":
            for field in ("package", "version"):
                if field not in pending:
                    errors.append(f"Pending install confirmation missing {field}")
        if pending.get("action") in {"run_local_commands", "prepare_venv", "apply_file_operations", "install_package"}:
            if not pending.get("command") and not pending.get("commands"):
                warnings.append(f"Pending action {pending.get('action')} has no command payload")
    return errors, warnings


def _validate_dataset(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    report = state.get("data_report", {})
    for key in ("shape", "columns"):
        if key not in report:
            errors.append(f"Dataset report missing {key}")
    shape = report.get("shape", [])
    if not isinstance(shape, list) or len(shape) < 2:
        errors.append("Dataset report shape must be [rows, cols]")
    else:
        try:
            if int(shape[0]) <= 0:
                errors.append("Dataset report has no rows")
        except Exception:
            errors.append("Dataset report shape[0] must be numeric")
    source = report.get("source", {}) if isinstance(report, dict) else {}
    if not isinstance(source, dict) or "requested_source" not in source or "resolved_source" not in source:
        warnings.append("Dataset report missing source resolution metadata")
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
    problem_type = str((state.get("research_plan") or {}).get("problem_type") or "").strip().lower()
    if problem_type and problem_type not in ALLOWED_PROBLEM_TYPES:
        errors.append(f"Codegen received unsupported problem_type: {problem_type}")
    code_level = str((state.get("research_plan") or {}).get("code_level") or "").strip().lower()
    if code_level and code_level not in ALLOWED_CODE_LEVELS:
        errors.append(f"Codegen received unsupported code_level: {code_level}")
    if is_vscode_execution_mode(state):
        planned = {str(item.get("path", "")) for item in state.get("local_file_plan", [])}
        for path in required:
            if str(path) not in planned:
                errors.append(f"Code generation missing planned file: {path.name}")
        config_entry = next((item for item in state.get("local_file_plan", []) if str(item.get("path", "")).endswith("config.py")), None)
        if isinstance(config_entry, dict):
            content = str(config_entry.get("content", ""))
            if "PROBLEM_TYPE" not in content:
                errors.append("Code generation config.py missing PROBLEM_TYPE")
            if "CODE_LEVEL" not in content:
                errors.append("Code generation config.py missing CODE_LEVEL")
    else:
        for path in required:
            if not path.exists():
                errors.append(f"Code generation missing file: {path.name}")
        config_file = project / "config.py"
        if config_file.exists():
            content = config_file.read_text(encoding="utf-8", errors="ignore")
            if "PROBLEM_TYPE" not in content:
                errors.append("Code generation config.py missing PROBLEM_TYPE")
            if "CODE_LEVEL" not in content:
                errors.append("Code generation config.py missing CODE_LEVEL")
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
        if "QUBIT_COUNT" not in code:
            warnings.append("Quantum circuit code missing QUBIT_COUNT constant")
        if "BACKEND" not in code:
            warnings.append("Quantum circuit code missing BACKEND constant")
        path = Path(state["project_path"]) / "src" / "quantum_circuit.py"
        if is_vscode_execution_mode(state):
            planned = {str(item.get("path", "")) for item in state.get("local_file_plan", [])}
            if str(path) not in planned:
                errors.append("Quantum gate did not plan src/quantum_circuit.py")
        elif not path.exists():
            errors.append("Quantum gate did not create src/quantum_circuit.py")
    return errors, warnings


def _validate_scheduler(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    order = state.get("execution_order", [])
    if not order:
        errors.append("Scheduler produced empty execution_order")
    if order and not any(str(item).endswith("main.py") for item in order):
        warnings.append("Scheduler execution_order does not include main.py")
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
    duration = _as_float(latest.get("duration_sec"))
    if duration is None or duration < 0:
        errors.append("Execution log duration_sec must be >= 0")
    if latest.get("returncode") is None:
        errors.append("Execution log missing returncode value")
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
    if state.get("retry_count", 0) > 0 and not state.get("repair_history"):
        warnings.append("Retry count > 0 but repair_history is empty")
    return errors, warnings


def _validate_evaluator(state: ResearchState) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not state.get("metrics"):
        errors.append("Evaluator did not populate metrics")
    summary = state.get("evaluation_summary", {})
    if "metrics" not in summary:
        warnings.append("Evaluation summary missing metrics section")
    evaluation = (state.get("metrics") or {}).get("evaluation", {})
    if not isinstance(evaluation, dict):
        errors.append("Evaluator metrics.evaluation must be an object")
        evaluation = {}
    primary_name = str(state.get("target_metric") or "accuracy")
    if primary_name not in evaluation:
        warnings.append(f"Primary target metric '{primary_name}' missing from evaluation map")
    else:
        if _as_float(evaluation.get(primary_name)) is None:
            errors.append(f"Primary target metric '{primary_name}' is not numeric")
    if bool(state.get("requires_quantum")) and "quantum_benchmarks" not in (state.get("metrics") or {}):
        warnings.append("Quantum run missing quantum_benchmarks in evaluator output")
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
    report_path = Path(doc_path)
    if is_vscode_execution_mode(state):
        pending = state.get("pending_user_confirm") or {}
        pending_write = False
        if str(pending.get("action", "")).strip().lower() == "apply_file_operations":
            for item in pending.get("file_operations", []) if isinstance(pending.get("file_operations"), list) else []:
                if str(item.get("path", "")).strip() == doc_path:
                    pending_write = True
                    break
            if not pending_write:
                pending_write = doc_path in {
                    str(item).strip()
                    for item in (pending.get("created_files", []) if isinstance(pending.get("created_files"), list) else [])
                    if str(item).strip()
                }
        local_materialized = {
            str(item).strip()
            for item in (state.get("local_materialized_files", []) if isinstance(state.get("local_materialized_files"), list) else [])
            if str(item).strip()
        }
        report_text = ""
        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        else:
            report_text = str(state.get("documentation_content") or "")
            if not report_text:
                for item in reversed(state.get("local_file_plan", []) if isinstance(state.get("local_file_plan"), list) else []):
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("path", "")).strip() != doc_path:
                        continue
                    report_text = str(item.get("content", ""))
                    if report_text:
                        break
        if not report_text.strip() and not pending_write and doc_path not in local_materialized:
            errors.append("final report file missing")
            return errors, warnings
        if not report_text.strip():
            warnings.append("Final report content not available for validation yet")
            return errors, warnings
    else:
        if not report_path.exists():
            errors.append("final report file missing")
            return errors, warnings
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    for section in ["# Abstract", "## Research Objective", "## Experimental Results", "## Conclusion & Interpretation"]:
        if section not in report_text:
            warnings.append(f"Final report missing section marker: {section}")
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
