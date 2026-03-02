from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypedDict
import time


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting_user"
    SUCCESS = "success"
    ABORTED = "aborted"
    FAILED = "failed"


class ErrorRecord(TypedDict):
    category: str
    message: str
    file_path: str
    line_number: int
    traceback: str
    timestamp: float


class ExecutionLog(TypedDict):
    script_path: str
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    timestamp: float
    executor: str
    host: str


class RepairRecord(TypedDict):
    attempt: int
    error_category: str
    fix_description: str
    file_changed: str
    find_text: str
    replace_text: str
    timestamp: float


class DenialRecord(TypedDict):
    action: str
    denied_item: str
    reason: str
    alternative_offered: str
    alternative_accepted: bool
    timestamp: float


class LocalFilePlanItem(TypedDict):
    path: str
    content: str
    phase: str


class LocalActionResult(TypedDict):
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    command: list[str]
    cwd: str
    created_files: list[str]
    metadata: dict[str, Any]


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in {0, 1}:
            return bool(value)
        return default
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled", "disable"}:
        return False
    return default


class ResearchState(TypedDict):
    experiment_id: str
    project_path: str

    phase: str
    status: ExperimentStatus
    timestamp_start: float
    timestamp_end: Optional[float]

    user_prompt: str
    research_type: str
    clarifications: dict[str, Any]
    research_plan: dict[str, Any]

    requires_quantum: bool
    quantum_framework: Optional[str]
    quantum_algorithm: Optional[str]
    quantum_qubit_count: Optional[int]
    quantum_circuit_code: Optional[str]
    quantum_backend: Optional[str]

    framework: str
    python_version: str
    required_packages: list[str]
    installed_packages: list[str]
    venv_path: str
    venv_ready: bool
    output_format: str

    dataset_source: str
    dataset_path: str
    kaggle_dataset_id: Optional[str]
    data_report: dict[str, Any]

    created_files: list[str]
    execution_order: list[str]

    execution_logs: list[ExecutionLog]
    current_script: Optional[str]
    total_duration_sec: Optional[float]

    errors: list[ErrorRecord]
    retry_count: int
    last_error_category: Optional[str]
    consecutive_same_error: int
    repair_history: list[RepairRecord]

    denied_actions: list[DenialRecord]
    pending_user_question: Optional[dict[str, Any]]
    pending_user_confirm: Optional[dict[str, Any]]

    metrics: dict[str, Any]
    plots_generated: list[str]
    evaluation_summary: dict[str, Any]

    documentation_path: Optional[str]
    documentation_content: Optional[str]
    report_sections: list[str]

    target_metric: str
    hardware_target: str
    random_seed: int
    max_epochs: Optional[int]
    batch_size: Optional[int]

    llm_calls_count: int
    total_tokens_used: int
    llm_total_cost_usd: float
    llm_provider: str
    llm_model: str
    execution_target: str
    execution_mode: str
    default_allow_research: bool
    confirmations_requested: int
    confirmations_processed: int
    phase_timings: dict[str, float]
    research_user_id: str
    test_mode: bool
    collection_key: str
    webhook_url: str
    local_file_plan: list[LocalFilePlanItem]
    local_materialized_files: list[str]
    local_action_history: list[dict[str, Any]]
    last_local_action_result: Optional[LocalActionResult]
    local_hardware_profile: dict[str, Any]
    local_python_command: str


def new_research_state(experiment_id: str, project_path: str, prompt: str, overrides: dict[str, Any]) -> ResearchState:
    resolved_project_path = str(Path(project_path).expanduser().resolve())
    resolved_project = Path(resolved_project_path)
    return ResearchState(
        experiment_id=experiment_id,
        project_path=resolved_project_path,
        phase="clarifier",
        status=ExperimentStatus.PENDING.value,
        timestamp_start=time.time(),
        timestamp_end=None,
        user_prompt=prompt,
        research_type=str(overrides.get("research_type", "ai")),
        clarifications={},
        research_plan={},
        requires_quantum=False,
        quantum_framework=None,
        quantum_algorithm=None,
        quantum_qubit_count=None,
        quantum_circuit_code=None,
        quantum_backend=None,
        framework="sklearn",
        python_version=str(overrides.get("python_version", "3.11")),
        required_packages=[],
        installed_packages=[],
        venv_path=str((resolved_project / ".venv").resolve()),
        venv_ready=False,
        output_format=str(overrides.get("output_format", ".py")),
        dataset_source=str(overrides.get("dataset_source", "sklearn")),
        dataset_path=str((resolved_project / "data" / "raw").resolve()),
        kaggle_dataset_id=overrides.get("kaggle_dataset_id"),
        data_report={},
        created_files=[],
        execution_order=[],
        execution_logs=[],
        current_script=None,
        total_duration_sec=None,
        errors=[],
        retry_count=0,
        last_error_category=None,
        consecutive_same_error=0,
        repair_history=[],
        denied_actions=[],
        pending_user_question=None,
        pending_user_confirm=None,
        metrics={},
        plots_generated=[],
        evaluation_summary={},
        documentation_path=None,
        documentation_content=None,
        report_sections=[],
        target_metric=str(overrides.get("target_metric", "accuracy")),
        hardware_target=str(overrides.get("hardware_target", "cpu")),
        random_seed=int(overrides.get("random_seed", 42)),
        max_epochs=int(overrides.get("max_epochs", 50)),
        batch_size=int(overrides.get("batch_size", 32)),
        llm_calls_count=0,
        total_tokens_used=0,
        llm_total_cost_usd=0.0,
        llm_provider="",
        llm_model="",
        execution_target="local_machine",
        execution_mode=str(overrides.get("execution_mode", "vscode_extension")),
        default_allow_research=_as_bool(overrides.get("default_allow_research", False), default=False),
        confirmations_requested=0,
        confirmations_processed=0,
        phase_timings={},
        research_user_id=str(overrides.get("user_id", "anonymous")),
        test_mode=bool(overrides.get("test_mode", False)),
        collection_key=str(overrides.get("collection_key", "user:anonymous")),
        webhook_url=str(overrides.get("webhook_url", "") or ""),
        local_file_plan=[],
        local_materialized_files=[],
        local_action_history=[],
        last_local_action_result=None,
        local_hardware_profile=dict(overrides.get("local_hardware_profile") or {}),
        local_python_command=str(overrides.get("local_python_command", "python")),
    )
