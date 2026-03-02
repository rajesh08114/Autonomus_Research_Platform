from __future__ import annotations

import json

import pytest

from src.agents.evaluator_agent import evaluator_agent_node
from src.agents.job_scheduler_agent import job_scheduler_agent_node
from src.state.research_state import ExperimentStatus, new_research_state


@pytest.mark.asyncio
async def test_job_scheduler_preflight_routes_to_error_recovery_on_syntax_error(tmp_path):
    state = new_research_state(
        "exp_scheduler_preflight",
        str(tmp_path),
        "run pipeline",
        {"execution_mode": "vscode_extension"},
    )
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "job_scheduler"
    state["data_report"] = {"shape": [50, 4], "columns": ["feature_1", "feature_2", "feature_3", "target"]}
    main_path = str((tmp_path / "main.py").resolve())
    state["local_file_plan"] = [{"path": main_path, "content": "def broken(:\n    pass\n", "phase": "code_generator"}]
    state["created_files"] = [main_path]

    updated = await job_scheduler_agent_node(state)
    assert updated["phase"] == "error_recovery"
    assert updated["errors"]
    assert updated["errors"][-1]["category"] == "SchedulerPreflightError"


@pytest.mark.asyncio
async def test_evaluator_uses_execution_log_metrics_when_metrics_file_missing(tmp_path):
    state = new_research_state(
        "exp_eval_log_metrics",
        str(tmp_path),
        "evaluate run",
        {"execution_mode": "vscode_extension"},
    )
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "results_evaluator"
    state["target_metric"] = "accuracy"
    state["execution_logs"] = [
        {
            "script_path": str((tmp_path / "main.py").resolve()),
            "command": ["python", "main.py"],
            "cwd": str(tmp_path),
            "returncode": 0,
            "stdout": "METRIC: accuracy=0.88\n",
            "stderr": "",
            "duration_sec": 2.4,
            "timestamp": 0.0,
            "executor": "python",
            "host": "test",
        }
    ]

    updated = await evaluator_agent_node(state)
    accuracy = float((updated.get("metrics", {}).get("evaluation", {}) or {}).get("accuracy", 0.0))
    assert accuracy == pytest.approx(0.88, rel=1e-6)


@pytest.mark.asyncio
async def test_job_scheduler_hybrid_mode_includes_notebook_runner(tmp_path):
    state = new_research_state(
        "exp_scheduler_hybrid",
        str(tmp_path),
        "run hybrid pipeline",
        {"execution_mode": "vscode_extension"},
    )
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "job_scheduler"
    state["output_format"] = "hybrid"
    state["data_report"] = {"shape": [30, 4], "columns": ["f1", "f2", "f3", "target"]}
    validate_path = str((tmp_path / "data" / "validate_data.py").resolve())
    runner_path = str((tmp_path / "notebooks" / "run_notebook.py").resolve())
    notebook_path = str((tmp_path / "notebooks" / "research_workflow.ipynb").resolve())
    main_path = str((tmp_path / "main.py").resolve())
    state["local_file_plan"] = [
        {"path": validate_path, "content": "print('ok')\n", "phase": "dataset_manager"},
        {"path": runner_path, "content": "print('run notebook')\n", "phase": "code_generator"},
        {
            "path": notebook_path,
            "content": json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "source": ["x = 1\n", "print(x)\n"],
                            "metadata": {},
                            "outputs": [],
                            "execution_count": None,
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            "phase": "code_generator",
        },
        {"path": main_path, "content": "print('main')\n", "phase": "code_generator"},
    ]
    state["created_files"] = [validate_path, runner_path, notebook_path, main_path]

    updated = await job_scheduler_agent_node(state)
    assert updated["phase"] == "job_scheduler"
    order = [str(item) for item in updated.get("execution_order", [])]
    assert any(path.endswith("notebooks\\run_notebook.py") or path.endswith("notebooks/run_notebook.py") for path in order)
    assert any(path.endswith("main.py") for path in order)


@pytest.mark.asyncio
async def test_evaluator_uses_notebook_result_metrics(tmp_path):
    state = new_research_state(
        "exp_eval_notebook_metrics",
        str(tmp_path),
        "evaluate notebook run",
        {"execution_mode": "vscode_extension"},
    )
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "results_evaluator"
    state["target_metric"] = "accuracy"
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "notebook_results.json").write_text(
        json.dumps(
            {
                "metrics": {"accuracy": 0.91, "f1": 0.89},
                "cells_executed": 4,
                "cells_failed": 0,
            }
        ),
        encoding="utf-8",
    )

    updated = await evaluator_agent_node(state)
    evaluation = (updated.get("metrics", {}).get("evaluation", {}) or {})
    assert float(evaluation.get("accuracy", 0.0)) == pytest.approx(0.91, rel=1e-6)
    assert "notebook_execution" in (updated.get("metrics") or {})
