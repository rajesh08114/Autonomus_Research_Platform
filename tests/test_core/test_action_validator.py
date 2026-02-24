from __future__ import annotations

from src.core.action_validator import validate_action
from src.state.research_state import ExperimentStatus, ResearchState


def _state() -> ResearchState:
    return ResearchState(
        experiment_id="exp_test",
        project_path="/tmp/project",
        phase="x",
        status=ExperimentStatus.RUNNING,
        timestamp_start=0.0,
        timestamp_end=None,
        user_prompt="x",
        clarifications={},
        research_plan={},
        requires_quantum=False,
        quantum_framework=None,
        quantum_algorithm=None,
        quantum_qubit_count=None,
        quantum_circuit_code=None,
        quantum_backend=None,
        framework="sklearn",
        python_version="3.11",
        required_packages=[],
        installed_packages=[],
        venv_path="",
        output_format=".py",
        dataset_source="synthetic",
        dataset_path="",
        kaggle_dataset_id=None,
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
        report_sections=[],
        target_metric="accuracy",
        hardware_target="cpu",
        random_seed=42,
        max_epochs=50,
        batch_size=32,
        llm_calls_count=0,
        total_tokens_used=0,
        phase_timings={},
    )


def test_validate_action_ok():
    state = _state()
    action = {
        "action": "run_python",
        "reasoning": "Execute training script in isolated process.",
        "next_step": "subprocess_runner",
        "parameters": {"script_path": "/tmp/project/main.py", "timeout_seconds": 60},
    }
    ok, _ = validate_action(action, state)
    assert ok
