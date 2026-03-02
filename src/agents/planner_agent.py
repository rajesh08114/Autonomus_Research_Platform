from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode
from src.core.local_actions import queue_local_file_action
from src.core.logger import get_logger
from src.core.user_behavior import build_user_behavior_profile
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disable", "disabled"}
_ALGORITHM_CLASS_OPTIONS = {"supervised", "unsupervised", "reinforcement", "quantum_ml"}
_DATASET_SOURCE_OPTIONS = {"kaggle", "sklearn", "synthetic", "upload"}
_OUTPUT_FORMAT_OPTIONS = {".py", ".ipynb", "hybrid"}
_HARDWARE_OPTIONS = {"cpu", "cuda", "ibmq"}
_FRAMEWORK_OPTIONS = {"auto", "sklearn", "pytorch"}
_PROBLEM_TYPE_OPTIONS = {"auto", "classification", "regression", "clustering", "reinforcement", "forecasting", "generation"}
_CODE_LEVEL_OPTIONS = {"low", "intermediate", "advanced"}
_PYTHON_OPTIONS = {"3.10", "3.11", "3.12"}
_QUANTUM_FRAMEWORK_OPTIONS = {"pennylane", "qiskit", "cirq"}
_QUANTUM_ALGORITHM_OPTIONS = {"VQE", "QAOA", "QNN", "Grover", "QSVM"}
_PACKAGE_PIN_PATTERN = r"^[a-zA-Z0-9_.-]+==[0-9][a-zA-Z0-9_.-]*$"


def _normalize_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _normalize_choice(value: Any, options: set[str], default: str) -> str:
    text = _normalize_text(value)
    if text in options:
        return text
    lower_map = {item.lower(): item for item in options}
    if text.lower() in lower_map:
        return lower_map[text.lower()]
    return default


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in {0, 1}:
            return bool(value)
        return None
    text = _normalize_text(value).lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _normalize_int(value: Any, default: int, minimum: int = 1, maximum: int = 10000) -> int:
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    else:
        text = _normalize_text(value)
        if not text:
            parsed = default
        else:
            try:
                parsed = int(float(text))
            except Exception:
                parsed = default
    return max(minimum, min(maximum, parsed))


def _resolve_quantum_algorithm(value: Any) -> str:
    raw = _normalize_text(value)
    if not raw:
        return "VQE"
    upper = raw.upper()
    if upper in _QUANTUM_ALGORITHM_OPTIONS:
        return upper
    if raw.lower() == "auto":
        return "VQE"
    return "VQE"


def _resolve_quantum_backend(value: Any, quantum_framework: str) -> str:
    requested = _normalize_text(value)
    if requested:
        return requested
    if quantum_framework == "qiskit":
        return "aer_simulator"
    if quantum_framework == "cirq":
        return "cirq-simulator"
    return "default.qubit"


def _infer_problem_type(clar: dict[str, Any], algorithm_class: str, target_metric: str) -> str:
    requested = _normalize_choice(clar.get("problem_type", "auto"), _PROBLEM_TYPE_OPTIONS, "auto")
    if requested != "auto":
        return requested
    metric = str(target_metric or "").strip().lower()
    if metric in {"rmse", "mae", "mse", "r2"}:
        return "regression"
    if algorithm_class == "unsupervised":
        return "clustering"
    if algorithm_class == "reinforcement":
        return "reinforcement"
    return "classification"


def _package_set_for_state(state: ResearchState) -> list[str]:
    packages = {
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib==3.8.4",
        "scikit-learn==1.5.2",
        "structlog==24.4.0",
    }
    framework = state["framework"]
    if framework == "pytorch":
        packages.update({"torch==2.5.1", "torchvision==0.20.1"})
    if state["requires_quantum"]:
        qf = state["quantum_framework"] or "pennylane"
        if qf == "pennylane":
            packages.update({"pennylane==0.37.0", "pennylane-lightning==0.37.0"})
        elif qf == "qiskit":
            packages.update({"qiskit==1.2.4", "qiskit-aer==0.15.1"})
        elif qf == "cirq":
            packages.add("cirq==1.3.0")
    if state["dataset_source"] == "kaggle":
        packages.add("kaggle==2.0.0")
    if _wants_notebook_artifacts(str(state.get("output_format") or "")):
        packages.update({"jupyter==1.1.1", "ipykernel==7.2.0"})
    return sorted(packages)


def _clean_text_list(value: Any, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _is_valid_pin(value: str) -> bool:
    return bool(re.match(_PACKAGE_PIN_PATTERN, value or ""))


def _wants_notebook_artifacts(output_format: str) -> bool:
    return str(output_format or "").strip().lower() in {".ipynb", "hybrid"}


def _build_base_plan(
    state: ResearchState,
    algorithm_class: str,
    problem_type: str,
    code_level: str,
    auto_retry_enabled: bool,
) -> dict[str, Any]:
    dataset_step = {
        "kaggle": "download dataset from Kaggle using configured dataset id",
        "sklearn": "load canonical sklearn dataset and persist to project data/raw",
        "synthetic": "generate synthetic dataset based on seed and task constraints",
        "upload": "ingest user-uploaded CSV from project data/raw",
    }.get(state["dataset_source"], "collect or generate dataset")
    training_step = "train hybrid quantum-classical model" if state["requires_quantum"] else "train baseline model"
    metrics_step = f"evaluate target metric ({state['target_metric']})"
    methodology = [
        dataset_step,
        "preprocess and split data",
        training_step,
        metrics_step,
        "document artifacts",
    ]
    if state["requires_quantum"]:
        methodology.insert(
            3,
            f"generate {state['quantum_framework']} circuit ({state['quantum_algorithm']}, {state['quantum_qubit_count']} qubits)",
        )
    if _wants_notebook_artifacts(str(state.get("output_format") or "")):
        methodology.append("export notebook-friendly artifacts")
    if str(state.get("output_format") or "").strip().lower() == "hybrid":
        methodology.append("maintain synchronized script + notebook outputs")

    algorithm_name = "classical_classifier"
    if state["requires_quantum"]:
        algorithm_name = f"{str(state.get('quantum_algorithm') or 'VQE').lower()}_hybrid_classifier"
    elif algorithm_class == "reinforcement":
        algorithm_name = "reinforcement_policy_model"
    elif algorithm_class == "unsupervised":
        algorithm_name = "unsupervised_pattern_model"
    elif problem_type == "regression":
        algorithm_name = "regression_model"
    elif problem_type == "forecasting":
        algorithm_name = "forecasting_model"
    elif problem_type == "generation":
        algorithm_name = "generation_model"

    return {
        "objective": state["user_prompt"],
        "methodology": methodology,
        "algorithm": algorithm_name,
        "algorithm_class": algorithm_class,
        "problem_type": problem_type,
        "code_level": code_level,
        "framework": state["framework"],
        "dataset": {
            "source": state["dataset_source"],
            "kaggle_dataset_id": state["kaggle_dataset_id"],
            "expected_shape": "tabular rows x features+target",
        },
        "metrics": [state["target_metric"], "train_loss", "duration_sec"],
        "hardware": {
            "target": state["hardware_target"],
            "default_target": "cpu",
        },
        "reproducibility": {
            "seed": state["random_seed"],
            "python_version": state["python_version"],
            "pins": True,
        },
        "library_profile": "latest_stable",
        "auto_retry_on_low_metric": auto_retry_enabled,
        "estimated_duration_minutes": 20 if state["requires_quantum"] else 15,
    }


def _planner_scaffold_markdown(state: ResearchState, plan: dict[str, Any]) -> str:
    methodology = plan.get("methodology", [])
    steps = [f"- {str(step)}" for step in methodology] if isinstance(methodology, list) else []
    packages = [f"- {str(spec)}" for spec in state.get("required_packages", [])[:40]]
    if not packages:
        packages = ["- package resolution pending"]
    return (
        "# Planned Workspace Scaffold\n\n"
        f"Experiment: `{state['experiment_id']}`\n\n"
        "## Objective\n\n"
        f"{str(plan.get('objective') or state.get('user_prompt') or '')}\n\n"
        "## Methodology\n\n"
        f"{chr(10).join(steps) if steps else '- methodology pending'}\n\n"
        "## Package Baseline\n\n"
        f"{chr(10).join(packages)}\n"
    )


def _planner_notebook_cell_plan(state: ResearchState) -> list[dict[str, Any]]:
    metric = str(state.get("target_metric") or "accuracy")
    return [
        {
            "cell_id": "cell_01_objective",
            "cell_type": "markdown",
            "purpose": "Capture objective and assumptions",
            "expects_output": False,
        },
        {
            "cell_id": "cell_02_imports_setup",
            "cell_type": "code",
            "purpose": "Import project modules and initialize reproducibility settings",
            "expects_output": True,
        },
        {
            "cell_id": "cell_03_data_loading",
            "cell_type": "code",
            "purpose": "Load and preprocess dataset from generated pipeline",
            "expects_output": True,
        },
        {
            "cell_id": "cell_04_training",
            "cell_type": "code",
            "purpose": "Train model and produce training artifacts",
            "expects_output": True,
        },
        {
            "cell_id": "cell_05_evaluation",
            "cell_type": "code",
            "purpose": f"Evaluate model and print primary metric ({metric})",
            "expects_output": True,
        },
        {
            "cell_id": "cell_06_persist_results",
            "cell_type": "code",
            "purpose": "Persist notebook execution summary to outputs/notebook_results.json",
            "expects_output": True,
        },
    ]


def _planner_notebook_stub(state: ResearchState) -> str:
    objective = str(state.get("user_prompt") or "").strip()
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# Experiment Notebook\n\nObjective: {objective}\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Notebook code cells are generated in code_generator phase.\n",
                    "print('Notebook scaffold ready')\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2)


def _planner_scaffold_operations(state: ResearchState, plan: dict[str, Any]) -> list[dict[str, str]]:
    project = Path(state["project_path"]).resolve()
    required_dirs = [
        "src",
        "data",
        "data/raw",
        "data/processed",
        "outputs",
        "outputs/plots",
        "outputs/model_checkpoint",
        "logs",
        "docs",
    ]
    if _wants_notebook_artifacts(str(state.get("output_format") or "")):
        required_dirs.append("notebooks")
    operations: list[dict[str, str]] = []
    for rel in required_dirs:
        operations.append(
            {
                "path": str((project / rel).resolve()),
                "mode": "mkdir",
                "phase": "planner",
            }
        )
    operations.append(
        {
            "path": str((project / "docs" / "plan.md").resolve()),
            "mode": "write",
            "content": _planner_scaffold_markdown(state, plan),
            "phase": "planner",
        }
    )
    for rel in (
        "config.py",
        "main.py",
        "src/__init__.py",
        "src/utils.py",
        "src/preprocessing.py",
        "src/model.py",
        "src/train.py",
        "src/evaluate.py",
    ):
        operations.append(
            {
                "path": str((project / rel).resolve()),
                "mode": "write",
                "content": "",
                "phase": "planner",
            }
        )
    if _wants_notebook_artifacts(str(state.get("output_format") or "")):
        operations.append(
            {
                "path": str((project / "notebooks" / "research_workflow.ipynb").resolve()),
                "mode": "write",
                "content": _planner_notebook_stub(state),
                "phase": "planner",
            }
        )
        operations.append(
            {
                "path": str((project / "notebooks" / "cell_plan.json").resolve()),
                "mode": "write",
                "content": json.dumps(_planner_notebook_cell_plan(state), indent=2),
                "phase": "planner",
            }
        )
    return operations


async def _invoke_dynamic_planner(
    state: ResearchState,
    base_plan: dict[str, Any],
    user_behavior_profile: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = (
        "SYSTEM ROLE: planner_dynamic_plan.\n"
        "Return JSON only with keys:\n"
        "- objective_refinement (string)\n"
        "- methodology (array of strings)\n"
        "- algorithm (string)\n"
        "- risk_checks (array of strings)\n"
        "- package_additions (array of pinned packages pkg==version)\n"
        "- training_strategy (string)\n"
        "- optimizer (string)\n"
        "- circuit_layers (integer)\n"
        "- encoding (string)\n"
        "- estimated_duration_minutes (integer)\n"
        "Respect user constraints in state context."
    )
    user_prompt = json.dumps(
        {
            "clarifications": state.get("clarifications", {}),
            "research_plan_base": base_plan,
            "framework": state.get("framework"),
            "dataset_source": state.get("dataset_source"),
            "requires_quantum": state.get("requires_quantum"),
            "quantum": {
                "framework": state.get("quantum_framework"),
                "algorithm": state.get("quantum_algorithm"),
                "qubits": state.get("quantum_qubit_count"),
                "backend": state.get("quantum_backend"),
            },
            "target_metric": state.get("target_metric"),
            "required_packages": state.get("required_packages", []),
            "local_runtime": {
                "python_command": state.get("local_python_command"),
                "hardware_profile": state.get("local_hardware_profile", {}),
            },
            "user_behavior_profile": user_behavior_profile,
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="planner",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = parse_json_object(raw)
    if not parsed:
        logger.warning("agent.planner.dynamic_parse_failed", experiment_id=state["experiment_id"])
    return parsed


def _sanitize_dynamic_plan(payload: dict[str, Any], base_plan: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    patch: dict[str, Any] = {}

    objective_refinement = str(payload.get("objective_refinement", "")).strip()
    if objective_refinement:
        patch["objective"] = objective_refinement[:500]

    methodology = _clean_text_list(payload.get("methodology"), limit=8)
    if methodology:
        patch["methodology"] = methodology
    else:
        violations.append("methodology must be a non-empty string list")

    algorithm = str(payload.get("algorithm", "")).strip()
    if algorithm:
        patch["algorithm"] = algorithm[:120]
    else:
        violations.append("algorithm must be a non-empty string")

    risk_checks = _clean_text_list(payload.get("risk_checks"), limit=6)
    if risk_checks:
        patch["risk_checks"] = risk_checks

    package_additions = _clean_text_list(payload.get("package_additions"), limit=6)
    invalid_pins = [pin for pin in package_additions if not _is_valid_pin(pin)]
    if invalid_pins:
        violations.append(f"invalid package pins: {invalid_pins}")
    patch["package_additions"] = [pin for pin in package_additions if _is_valid_pin(pin)]

    training_strategy = str(payload.get("training_strategy", "")).strip()
    if training_strategy:
        patch["training_strategy"] = training_strategy[:120]
    optimizer = str(payload.get("optimizer", "")).strip()
    if optimizer:
        patch["optimizer"] = optimizer[:80]
    encoding = str(payload.get("encoding", "")).strip()
    if encoding:
        patch["encoding"] = encoding[:80]
    if "circuit_layers" in payload:
        patch["circuit_layers"] = _normalize_int(payload.get("circuit_layers"), default=3, minimum=1, maximum=20)
    if "estimated_duration_minutes" in payload:
        patch["estimated_duration_minutes"] = _normalize_int(
            payload.get("estimated_duration_minutes"),
            default=int(base_plan.get("estimated_duration_minutes", 15)),
            minimum=1,
            maximum=600,
        )
    return patch, violations


async def planner_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "planner"
    logger.info("agent.planner.start", experiment_id=state["experiment_id"])
    clar = state.get("clarifications", {})
    research_type = _normalize_choice(state.get("research_type", "ai"), {"ai", "quantum"}, "ai")
    algorithm_class = _normalize_choice(clar.get("algorithm_class", "supervised"), _ALGORITHM_CLASS_OPTIONS, "supervised")
    requires_quantum_decision = _normalize_bool(clar.get("requires_quantum"))
    if research_type == "quantum":
        state["requires_quantum"] = True
    elif requires_quantum_decision is not None:
        state["requires_quantum"] = bool(requires_quantum_decision)
    else:
        state["requires_quantum"] = bool(state.get("requires_quantum", False))

    selected_quantum_framework = _normalize_text(clar.get("quantum_framework", ""))
    if selected_quantum_framework.lower() == "no_preference":
        selected_quantum_framework = ""
    if state["requires_quantum"]:
        state["quantum_framework"] = _normalize_choice(
            selected_quantum_framework,
            _QUANTUM_FRAMEWORK_OPTIONS,
            "pennylane",
        )
        state["quantum_algorithm"] = _resolve_quantum_algorithm(clar.get("quantum_algorithm"))
        state["quantum_qubit_count"] = _normalize_int(clar.get("quantum_qubit_count"), default=4, minimum=1, maximum=64)
        state["quantum_backend"] = _resolve_quantum_backend(clar.get("quantum_backend"), state["quantum_framework"] or "pennylane")
    else:
        state["quantum_framework"] = None
        state["quantum_algorithm"] = None
        state["quantum_qubit_count"] = None
        state["quantum_backend"] = None

    state["dataset_source"] = _normalize_choice(clar.get("dataset_source", "sklearn"), _DATASET_SOURCE_OPTIONS, "sklearn")
    state["output_format"] = _normalize_choice(clar.get("output_format", ".py"), _OUTPUT_FORMAT_OPTIONS, ".py")
    target_metric_default = "fidelity" if state["requires_quantum"] else "accuracy"
    state["target_metric"] = _normalize_text(clar.get("target_metric", target_metric_default), target_metric_default)
    state["hardware_target"] = _normalize_choice(clar.get("hardware_target", "cpu"), _HARDWARE_OPTIONS, "cpu")
    problem_type = _infer_problem_type(clar, algorithm_class, state["target_metric"])
    code_level = _normalize_choice(clar.get("code_level", "intermediate"), _CODE_LEVEL_OPTIONS, "intermediate")
    clar["problem_type"] = problem_type
    clar["code_level"] = code_level
    state["random_seed"] = _normalize_int(clar.get("random_seed"), default=42, minimum=0, maximum=2147483647)
    state["max_epochs"] = _normalize_int(clar.get("max_epochs"), default=50, minimum=1, maximum=10000)
    state["python_version"] = _normalize_choice(clar.get("python_version", "3.11"), _PYTHON_OPTIONS, "3.11")
    state["kaggle_dataset_id"] = _normalize_text(clar.get("kaggle_dataset_id"), "")
    if state["dataset_source"] != "kaggle":
        state["kaggle_dataset_id"] = None

    framework_preference = _normalize_choice(clar.get("framework_preference", "auto"), _FRAMEWORK_OPTIONS, "auto")
    if framework_preference == "auto":
        if state["requires_quantum"] or algorithm_class == "reinforcement":
            state["framework"] = "pytorch"
        else:
            state["framework"] = "sklearn"
    else:
        state["framework"] = framework_preference

    auto_retry_enabled = _normalize_bool(clar.get("auto_retry_preference"))
    if auto_retry_enabled is None:
        auto_retry_enabled = True
    clar["auto_retry_preference"] = "enabled" if auto_retry_enabled else "disabled"
    state["clarifications"] = clar

    plan = _build_base_plan(
        state=state,
        algorithm_class=algorithm_class,
        problem_type=problem_type,
        code_level=code_level,
        auto_retry_enabled=auto_retry_enabled,
    )
    state["required_packages"] = _package_set_for_state(state)
    user_behavior_profile = build_user_behavior_profile(state)

    used_dynamic = False
    fallback_static = False
    dynamic_payload_keys: list[str] = []
    try:
        dynamic_payload = await _invoke_dynamic_planner(state, plan, user_behavior_profile)
        if not dynamic_payload:
            if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                fallback_static = True
                logger.warning("agent.planner.dynamic_fallback_static", experiment_id=state["experiment_id"], reason="parse_failed")
            else:
                raise RuntimeError("Planner dynamic plan parse failed")
        else:
            patch, violations = _sanitize_dynamic_plan(dynamic_payload, plan)
            dynamic_payload_keys = sorted(list(dynamic_payload.keys()))
            if violations:
                logger.warning("agent.planner.dynamic_validation_failed", experiment_id=state["experiment_id"], violations=violations)
                if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                    fallback_static = True
                    logger.warning(
                        "agent.planner.dynamic_fallback_static",
                        experiment_id=state["experiment_id"],
                        reason="validation_failed",
                    )
                else:
                    raise RuntimeError(f"Planner dynamic plan validation failed: {violations}")
            else:
                if "objective" in patch:
                    plan["objective"] = patch["objective"]
                if "methodology" in patch:
                    plan["methodology"] = patch["methodology"]
                if "algorithm" in patch:
                    plan["algorithm"] = patch["algorithm"]
                if "risk_checks" in patch:
                    plan["risk_checks"] = patch["risk_checks"]
                if "training_strategy" in patch:
                    plan["training_strategy"] = patch["training_strategy"]
                if "optimizer" in patch:
                    plan["optimizer"] = patch["optimizer"]
                if "encoding" in patch:
                    plan["encoding"] = patch["encoding"]
                if "circuit_layers" in patch:
                    plan["circuit_layers"] = patch["circuit_layers"]
                if "estimated_duration_minutes" in patch:
                    plan["estimated_duration_minutes"] = patch["estimated_duration_minutes"]
                additions = patch.get("package_additions", [])
                if isinstance(additions, list) and additions:
                    merged = set(state["required_packages"])
                    merged.update(additions)
                    state["required_packages"] = sorted(merged)
                used_dynamic = True
    except Exception as exc:
        logger.exception("agent.planner.dynamic_failed", experiment_id=state["experiment_id"])
        raise RuntimeError(f"Planner dynamic generation failed: {exc}") from exc

    state["research_plan"] = plan
    state["research_plan"]["planner_dynamic_summary"] = {
        "used_dynamic": used_dynamic,
        "fallback_static": fallback_static,
        "dynamic_payload_keys": dynamic_payload_keys,
        "package_count": len(state["required_packages"]),
        "user_decision_style": (user_behavior_profile.get("interaction") or {}).get("decision_style"),
    }

    scaffold_operations = _planner_scaffold_operations(state, plan)
    if is_vscode_execution_mode(state):
        queued = queue_local_file_action(
            state=state,
            phase="planner",
            file_operations=scaffold_operations,
            next_phase="env_manager",
            reason="Create initial project folders and planning notes locally before environment setup",
            cwd=state["project_path"],
        )
        state["research_plan"]["planner_dynamic_summary"]["scaffold_queued"] = bool(queued)
        state["research_plan"]["planner_dynamic_summary"]["scaffold_operation_count"] = len(scaffold_operations)
        if queued:
            logger.info(
                "agent.planner.pending_local_scaffold",
                experiment_id=state["experiment_id"],
                operation_count=len(scaffold_operations),
                next_phase="env_manager",
            )
            return state
    else:
        for op in scaffold_operations:
            mode = str(op.get("mode", "write")).strip().lower()
            target = Path(str(op.get("path", ""))).resolve()
            if mode == "mkdir":
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(op.get("content", "")), encoding="utf-8")
            path_text = str(target)
            if path_text not in state["created_files"]:
                state["created_files"].append(path_text)
        state["research_plan"]["planner_dynamic_summary"]["scaffold_queued"] = False
        state["research_plan"]["planner_dynamic_summary"]["scaffold_operation_count"] = len(scaffold_operations)

    state["status"] = ExperimentStatus.RUNNING.value
    logger.info(
        "agent.planner.end",
        experiment_id=state["experiment_id"],
        framework=state["framework"],
        requires_quantum=state["requires_quantum"],
        algorithm=state["research_plan"].get("algorithm"),
        package_count=len(state["required_packages"]),
    )
    return state
