from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents import code_gen_agent, dataset_agent, doc_generator_agent, env_manager_agent
from src.agents.code_gen_agent import code_gen_agent_node
from src.agents.dataset_agent import dataset_agent_node
from src.agents.doc_generator_agent import doc_generator_agent_node
from src.agents.env_manager_agent import env_manager_agent_node
from src.config.settings import settings
from src.state.research_state import ExperimentStatus, new_research_state


def _codegen_payload(problem_type: str, code_level: str, algorithm_class: str, extra_import: str = "") -> str:
    extra_line = f"import {extra_import}\n" if extra_import else ""
    payload = {
        "problem_type": problem_type,
        "code_level": code_level,
        "algorithm_class": algorithm_class,
        "files": [
            {
                "path": "config.py",
                "content": (
                    "from __future__ import annotations\n"
                    "import random\n"
                    f"PROBLEM_TYPE = '{problem_type}'\n"
                    f"CODE_LEVEL = '{code_level}'\n"
                    f"ALGORITHM_CLASS = '{algorithm_class}'\n"
                    "TARGET_METRIC = 'accuracy'\n"
                    "DEVICE = 'cpu'\n"
                    "def set_global_seed() -> None:\n"
                    "    random.seed(42)\n"
                ),
            },
            {"path": "main.py", "content": "from src.train import run_training\n\nrun_training()\n"},
            {"path": "src/__init__.py", "content": ""},
            {"path": "src/utils.py", "content": f"from __future__ import annotations\n{extra_line}def ping() -> str:\n    return 'ok'\n"},
            {"path": "src/preprocessing.py", "content": "from __future__ import annotations\n\n"},
            {"path": "src/model.py", "content": "from __future__ import annotations\n\n"},
            {"path": "src/train.py", "content": "from __future__ import annotations\n\ndef run_training() -> dict:\n    return {}\n"},
            {"path": "src/evaluate.py", "content": "from __future__ import annotations\n\n"},
        ],
    }
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_codegen_strict_repair_removes_disallowed_import(tmp_path):
    state = new_research_state("exp_codegen_repair", str(tmp_path), "build classifier", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value

    updated = await code_gen_agent_node(state)
    assert updated["status"] == ExperimentStatus.WAITING.value
    assert updated["pending_user_confirm"] is not None
    assert updated["pending_user_confirm"]["action"] == "apply_file_operations"
    assert updated["llm_calls_count"] >= 2
    utils_entry = next(item for item in updated["local_file_plan"] if item["path"].endswith("src\\utils.py") or item["path"].endswith("src/utils.py"))
    assert "forbiddenlib" not in utils_entry["content"]
    assert (updated.get("research_plan") or {}).get("codegen_strict_violations") == []


@pytest.mark.asyncio
async def test_codegen_strict_hard_fail_after_repair_attempt(tmp_path, monkeypatch):
    async def _always_invalid(*args, **kwargs) -> str:
        _ = (args, kwargs)
        return _codegen_payload("classification", "intermediate", "supervised", extra_import="forbiddenlib")

    monkeypatch.setattr(code_gen_agent, "invoke_master_llm", _always_invalid)
    state = new_research_state("exp_codegen_fail", str(tmp_path), "build classifier", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value

    with pytest.raises(RuntimeError, match="strict_state_only"):
        await code_gen_agent_node(state)


@pytest.mark.asyncio
async def test_codegen_strict_algorithm_mismatch_rejected(tmp_path, monkeypatch):
    async def _bad_algo(*args, **kwargs) -> str:
        _ = (args, kwargs)
        return _codegen_payload("classification", "intermediate", "unsupervised")

    monkeypatch.setattr(code_gen_agent, "invoke_master_llm", _bad_algo)
    state = new_research_state("exp_codegen_algo", str(tmp_path), "build classifier", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value
    state["research_plan"]["algorithm_class"] = "supervised"

    with pytest.raises(RuntimeError, match="algorithm_class mismatch"):
        await code_gen_agent_node(state)


@pytest.mark.asyncio
async def test_dataset_dynamic_success_with_contract(tmp_path):
    state = new_research_state("exp_dataset_dynamic", str(tmp_path), "prepare dataset", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value

    updated = await dataset_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("dataset_dynamic_plan_summary", {})
    assert summary.get("used_dynamic_plan") is True
    assert summary.get("fallback_static") is False
    assert "shape" in updated["data_report"]
    assert "columns" in updated["data_report"]
    assert "source" in updated["data_report"]
    assert updated["pending_user_confirm"]["action"] == "apply_file_operations"


@pytest.mark.asyncio
async def test_dataset_dynamic_fallback_to_static(tmp_path, monkeypatch):
    async def _invalid_plan(*args, **kwargs) -> str:
        _ = (args, kwargs)
        return "not json"

    monkeypatch.setattr(dataset_agent, "invoke_master_llm", _invalid_plan)
    state = new_research_state("exp_dataset_fallback", str(tmp_path), "prepare dataset", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value

    updated = await dataset_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("dataset_dynamic_plan_summary", {})
    assert summary.get("used_dynamic_plan") is False
    assert summary.get("fallback_static") is True
    assert "shape" in updated["data_report"]
    assert "columns" in updated["data_report"]


@pytest.mark.asyncio
async def test_env_dynamic_success_preserves_confirmation_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EXPERIMENT_VENV_ENABLED", False)
    state = new_research_state("exp_env_dynamic", str(tmp_path), "setup env", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value
    state["required_packages"] = ["numpy==1.26.4", "pandas==2.2.2"]

    updated = await env_manager_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("env_dynamic_plan_summary", {})
    assert summary.get("used_dynamic") is True
    assert summary.get("selected_package") in {"numpy==1.26.4", "pandas==2.2.2"}
    assert updated["pending_user_confirm"]["action"] == "install_package"
    assert updated["pending_user_confirm"]["phase"] == "env_manager"


@pytest.mark.asyncio
async def test_env_dynamic_invalid_plan_falls_back_to_static(tmp_path, monkeypatch):
    async def _invalid_recommendation(*args, **kwargs) -> str:
        _ = (args, kwargs)
        return json.dumps({"action": "install_package", "package_spec": "unknown==0.0.1", "reason": "invalid"})

    monkeypatch.setattr(settings, "EXPERIMENT_VENV_ENABLED", False)
    monkeypatch.setattr(env_manager_agent, "invoke_master_llm", _invalid_recommendation)
    state = new_research_state("exp_env_fallback", str(tmp_path), "setup env", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value
    state["required_packages"] = ["numpy==1.26.4", "pandas==2.2.2"]

    updated = await env_manager_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("env_dynamic_plan_summary", {})
    assert summary.get("fallback_static") is True
    assert summary.get("selected_package") == "numpy==1.26.4"
    assert updated["pending_user_confirm"]["action"] == "install_package"
    assert updated["pending_user_confirm"]["command"][:4] == ["python", "-m", "pip", "install"]


@pytest.mark.asyncio
async def test_doc_dynamic_success_writes_required_markers(tmp_path):
    state = new_research_state("exp_doc_dynamic", str(tmp_path), "write docs", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value
    state["metrics"] = {"evaluation": {"accuracy": 0.9}}

    updated = await doc_generator_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("doc_generation_summary", {})
    assert summary.get("used_dynamic") is True
    assert summary.get("fallback_static") is False
    report_text = Path(updated["documentation_path"]).read_text(encoding="utf-8")
    assert "# Abstract" in report_text
    assert "## Research Objective" in report_text
    assert "## Experimental Results" in report_text
    assert "## Conclusion & Interpretation" in report_text


@pytest.mark.asyncio
async def test_doc_dynamic_fallback_to_legacy_builder(tmp_path, monkeypatch):
    async def _raise_doc_error(*args, **kwargs) -> str:
        _ = (args, kwargs)
        raise RuntimeError("llm failed")

    monkeypatch.setattr(doc_generator_agent, "invoke_master_llm", _raise_doc_error)
    state = new_research_state("exp_doc_fallback", str(tmp_path), "write docs", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value
    state["metrics"] = {"evaluation": {"accuracy": 0.9}}

    updated = await doc_generator_agent_node(state)
    summary = (updated.get("research_plan") or {}).get("doc_generation_summary", {})
    assert summary.get("fallback_static") is True
    report_text = Path(updated["documentation_path"]).read_text(encoding="utf-8")
    assert "# Abstract" in report_text
