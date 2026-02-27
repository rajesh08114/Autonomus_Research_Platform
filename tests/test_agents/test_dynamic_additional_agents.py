from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents import job_scheduler_agent
from src.agents.evaluator_agent import evaluator_agent_node
from src.agents.job_scheduler_agent import job_scheduler_agent_node
from src.agents.planner_agent import planner_agent_node
from src.db.repository import ExperimentRepository
from src.state.research_state import ExperimentStatus, new_research_state


@pytest.mark.asyncio
async def test_planner_dynamic_success(tmp_path):
    state = new_research_state("exp_planner_dynamic", str(tmp_path), "build a classifier", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value

    updated = await planner_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("planner_dynamic_summary", {})
    assert summary.get("used_dynamic") is True
    assert summary.get("fallback_static") is False
    assert "seaborn==0.13.2" in updated.get("required_packages", [])
    assert isinstance(updated["research_plan"].get("methodology"), list) and updated["research_plan"]["methodology"]


@pytest.mark.asyncio
async def test_job_scheduler_dynamic_success_includes_main(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "validate_data.py").write_text("print('validate')\n", encoding="utf-8")

    state = new_research_state("exp_scheduler_dynamic", str(tmp_path), "schedule jobs", {})
    state["status"] = ExperimentStatus.RUNNING.value
    state["research_plan"] = {"code_level": "advanced", "algorithm_class": "supervised"}

    updated = await job_scheduler_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("scheduler_dynamic_plan_summary", {})
    assert summary.get("used_dynamic") is True
    main_path = str((Path(str(tmp_path)) / "main.py").resolve())
    assert main_path in updated["execution_order"]


@pytest.mark.asyncio
async def test_job_scheduler_dynamic_invalid_payload_falls_back(tmp_path, monkeypatch):
    async def _invalid_scheduler(*args, **kwargs) -> str:
        _ = (args, kwargs)
        return json.dumps({"execution_order": ["/tmp/outside_project.py"], "rationale": "invalid"})

    monkeypatch.setattr(job_scheduler_agent, "invoke_master_llm", _invalid_scheduler)
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "validate_data.py").write_text("print('validate')\n", encoding="utf-8")

    state = new_research_state("exp_scheduler_fallback", str(tmp_path), "schedule jobs", {})
    state["status"] = ExperimentStatus.RUNNING.value
    state["research_plan"] = {"code_level": "advanced", "algorithm_class": "supervised"}

    updated = await job_scheduler_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("scheduler_dynamic_plan_summary", {})
    assert summary.get("fallback_static") is True
    assert updated["execution_order"]


@pytest.mark.asyncio
async def test_evaluator_dynamic_interpretation_success(tmp_path, monkeypatch):
    async def _noop_add_metrics_snapshot(*args, **kwargs):
        _ = (args, kwargs)
        return None

    monkeypatch.setattr(ExperimentRepository, "add_metrics_snapshot", _noop_add_metrics_snapshot)
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_eval",
                "algorithm": "stub_model",
                "framework": "sklearn",
                "dataset": "synthetic",
                "training": {"duration_sec": 1.2, "final_loss": 0.1, "epochs": 5},
                "evaluation": {"accuracy": 0.91},
                "artifacts": {"plots": []},
            }
        ),
        encoding="utf-8",
    )

    state = new_research_state("exp_eval_dynamic", str(tmp_path), "evaluate", {})
    state["status"] = ExperimentStatus.RUNNING.value
    state["target_metric"] = "accuracy"

    updated = await evaluator_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("evaluator_dynamic_summary", {})
    interpretation = (updated.get("evaluation_summary") or {}).get("dynamic_interpretation", {})
    assert summary.get("used_dynamic") is True
    assert isinstance(interpretation.get("insights"), list) and interpretation["insights"]

