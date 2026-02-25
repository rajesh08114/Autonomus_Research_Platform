from __future__ import annotations

import json
from typing import Any

from src.core.logger import get_logger
from src.llm.master_llm import invoke_master_llm
from src.llm.response_parser import parse_json_response
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disable", "disabled"}
_ALGORITHM_CLASS_OPTIONS = {"supervised", "unsupervised", "reinforcement", "quantum_ml"}
_DATASET_SOURCE_OPTIONS = {"kaggle", "sklearn", "synthetic", "upload"}
_OUTPUT_FORMAT_OPTIONS = {".py", ".ipynb"}
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
        "numpy==2.4.2",
        "pandas==3.0.0",
        "matplotlib==3.10.3",
        "scikit-learn==1.8.0",
        "structlog==25.5.0",
    }
    framework = state["framework"]
    if framework == "pytorch":
        packages.update({"torch==2.10.0", "torchvision==0.25.0"})
    if state["requires_quantum"]:
        qf = state["quantum_framework"] or "pennylane"
        if qf == "pennylane":
            packages.update({"pennylane==0.43.0", "pennylane-lightning==0.43.0"})
        elif qf == "qiskit":
            packages.update({"qiskit==2.3.0", "qiskit-aer==0.17.2"})
        elif qf == "cirq":
            packages.add("cirq==1.6.1")
    if state["dataset_source"] == "kaggle":
        packages.add("kaggle==2.0.0")
    if state["output_format"] == ".ipynb":
        packages.update({"jupyter==1.1.1", "ipykernel==7.2.0"})
    return sorted(packages)


def _safe_parse(raw: str) -> dict[str, Any]:
    try:
        return parse_json_response(raw)
    except Exception:
        try:
            return json.loads(str(raw or "").strip())
        except Exception:
            return {}


def _clean_text_list(value: Any, limit: int = 4) -> list[str]:
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
    import re

    return bool(re.match(_PACKAGE_PIN_PATTERN, value or ""))


async def _llm_refine_plan(state: ResearchState, plan: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "objective": plan.get("objective"),
        "algorithm": plan.get("algorithm"),
        "framework": plan.get("framework"),
        "problem_type": plan.get("problem_type"),
        "code_level": plan.get("code_level"),
        "dataset_source": ((plan.get("dataset") or {}).get("source") if isinstance(plan.get("dataset"), dict) else None),
        "target_metric": state.get("target_metric"),
        "requires_quantum": bool(state.get("requires_quantum")),
    }
    system_prompt = (
        "You are a strict ML/AI planning reviewer. Return only one JSON object with keys: "
        "methodology_additions (list), risk_checks (list), package_additions (list of strict pinned "
        "pkg==version), algorithm_override (string or empty). Keep each list <= 3."
    )
    user_prompt = (
        f"Base plan context:\n{json.dumps(payload, indent=2, default=str)}\n\n"
        "Focus on latest stable APIs, reproducibility, and measurable evaluation quality."
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="planner",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = _safe_parse(raw)
    return parsed if isinstance(parsed, dict) else {}


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
        methodology.insert(3, f"generate {state['quantum_framework']} circuit ({state['quantum_algorithm']}, {state['quantum_qubit_count']} qubits)")
    if state["output_format"] == ".ipynb":
        methodology.append("export notebook-friendly artifacts")

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

    state["research_plan"] = {
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
            "fallback": "cpu",
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
    state["required_packages"] = _package_set_for_state(state)
    try:
        llm_patch = await _llm_refine_plan(state, state["research_plan"])
        if llm_patch:
            additions = _clean_text_list(llm_patch.get("methodology_additions"), limit=3)
            for item in additions:
                if item not in state["research_plan"]["methodology"]:
                    state["research_plan"]["methodology"].append(item)

            risk_checks = _clean_text_list(llm_patch.get("risk_checks"), limit=3)
            if risk_checks:
                state["research_plan"]["risk_checks"] = risk_checks

            algorithm_override = str(llm_patch.get("algorithm_override", "")).strip()
            if algorithm_override:
                state["research_plan"]["algorithm"] = algorithm_override

            package_additions = _clean_text_list(llm_patch.get("package_additions"), limit=3)
            if package_additions:
                merged = set(state["required_packages"])
                for pin in package_additions:
                    if _is_valid_pin(pin):
                        merged.add(pin)
                state["required_packages"] = sorted(merged)
    except Exception:
        logger.exception("agent.planner.llm_refine_failed", experiment_id=state["experiment_id"])
        if not settings.ALLOW_RULE_BASED_FALLBACK:
            raise RuntimeError("Planner LLM refinement failed and rule-based fallback is disabled.")
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
