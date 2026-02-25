from __future__ import annotations

import json
import re
from typing import Any

from src.config.settings import settings
from src.core.logger import get_logger
from src.llm.master_llm import invoke_master_llm
from src.llm.quantum_llm import invoke_quantum_llm_json
from src.llm.response_parser import parse_json_response
from src.prompts.clarifier import SYSTEM_PROMPT
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

MAX_DYNAMIC_QUESTIONS = 12
_ALLOWED_TYPES = {"choice", "text", "boolean", "number"}
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disable", "disabled"}

_QUESTION_LIBRARY: dict[str, dict[str, Any]] = {
    "output_format": {
        "text": "Do you want .py scripts or .ipynb notebooks?",
        "type": "choice",
        "options": [".py", ".ipynb"],
        "default": ".py",
        "required": True,
    },
    "algorithm_class": {
        "text": "What type of ML/AI task is this?",
        "type": "choice",
        "options": ["supervised", "unsupervised", "reinforcement", "quantum_ml"],
        "default": "supervised",
        "required": True,
    },
    "requires_quantum": {
        "text": "Does this experiment require quantum circuits?",
        "type": "boolean",
        "default": False,
        "required": True,
    },
    "quantum_framework": {
        "text": "Which quantum framework should be used?",
        "type": "choice",
        "options": ["pennylane", "qiskit", "cirq", "no_preference"],
        "default": "pennylane",
        "required": True,
    },
    "quantum_algorithm": {
        "text": "Which quantum algorithm style should be used?",
        "type": "choice",
        "options": ["VQE", "QAOA", "QNN", "Grover", "QSVM", "auto"],
        "default": "auto",
        "required": True,
    },
    "quantum_qubit_count": {
        "text": "How many qubits should be allocated?",
        "type": "number",
        "default": 4,
        "required": False,
    },
    "quantum_backend": {
        "text": "Which quantum backend should be targeted?",
        "type": "choice",
        "options": ["default.qubit", "lightning.qubit", "aer_simulator", "ibmq_qasm_simulator"],
        "default": "default.qubit",
        "required": False,
    },
    "dataset_source": {
        "text": "Where should the dataset come from?",
        "type": "choice",
        "options": ["kaggle", "sklearn", "synthetic", "upload"],
        "default": "sklearn",
        "required": True,
    },
    "kaggle_dataset_id": {
        "text": "Enter the Kaggle dataset path (username/dataset-name):",
        "type": "text",
        "default": "",
        "required": True,
    },
    "target_metric": {
        "text": "What is the primary evaluation metric?",
        "type": "choice",
        "options": ["accuracy", "f1_macro", "rmse", "roc_auc", "fidelity", "mae", "mse", "r2"],
        "default": "accuracy",
        "required": True,
    },
    "problem_type": {
        "text": "What problem type should the generated code target?",
        "type": "choice",
        "options": ["auto", "classification", "regression", "clustering", "reinforcement", "forecasting", "generation"],
        "default": "auto",
        "required": False,
    },
    "code_level": {
        "text": "Preferred code complexity level?",
        "type": "choice",
        "options": ["low", "intermediate", "advanced"],
        "default": "intermediate",
        "required": False,
    },
    "hardware_target": {
        "text": "Target hardware for execution?",
        "type": "choice",
        "options": ["cpu", "cuda", "ibmq"],
        "default": "cpu",
        "required": True,
    },
    "framework_preference": {
        "text": "Preferred training framework?",
        "type": "choice",
        "options": ["auto", "sklearn", "pytorch"],
        "default": "auto",
        "required": False,
    },
    "python_version": {
        "text": "Python version preference?",
        "type": "choice",
        "options": ["3.10", "3.11", "3.12"],
        "default": "3.11",
        "required": False,
    },
    "random_seed": {
        "text": "Random seed for reproducibility?",
        "type": "number",
        "default": 42,
        "required": False,
    },
    "max_epochs": {
        "text": "Maximum training epochs or circuit iterations?",
        "type": "number",
        "default": 50,
        "required": False,
    },
    "auto_retry_preference": {
        "text": "If metric is low, should the system auto-retry with longer training?",
        "type": "choice",
        "options": ["enabled", "disabled"],
        "default": "enabled",
        "required": False,
    },
}


def _infer_prompt_flags(prompt: str) -> dict[str, bool]:
    lower = prompt.lower()
    return {
        "requires_quantum_hint": any(token in lower for token in ["quantum", "vqe", "qaoa", "qnn", "circuit"]),
        "wants_notebook_hint": any(token in lower for token in ["notebook", "jupyter", ".ipynb"]),
        "wants_cuda_hint": "cuda" in lower or "gpu" in lower,
        "kaggle_hint": "kaggle" in lower,
        "wants_low_code_hint": any(token in lower for token in ["simple", "basic", "beginner", "easy"]),
        "wants_advanced_code_hint": any(token in lower for token in ["advanced", "production", "enterprise", "optimized"]),
    }


def _normalize_research_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"ai", "quantum"}:
        return normalized
    return "ai"


def _normalize_topic(value: str) -> str:
    return str(value or "").strip().lower()


def _question_id(topic: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _normalize_topic(topic)).strip("_")
    if slug:
        return f"Q_{slug.upper()}"
    return f"Q{index}"


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in {0, 1}:
            return bool(value)
        return None
    text = str(value or "").strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _normalize_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _normalize_choice(value: Any, options: list[str], default: Any = None) -> Any:
    text = str(value).strip()
    if text and text in options:
        return text
    lower_map = {opt.lower(): opt for opt in options}
    normalized = text.lower()
    if normalized in lower_map:
        return lower_map[normalized]
    if options:
        if default in options:
            return default
        return options[0]
    return value


def _wants_quantum(
    research_type: str,
    prompt_flags: dict[str, bool],
    clarifications: dict[str, Any],
) -> bool:
    if research_type == "quantum":
        return True
    decision = _normalize_bool(clarifications.get("requires_quantum"))
    if decision is not None:
        return decision
    return bool(prompt_flags.get("requires_quantum_hint"))


def _question_from_topic(
    topic: str,
    index: int,
    prompt_flags: dict[str, bool],
    research_type: str,
    clarifications: dict[str, Any],
) -> dict[str, Any]:
    spec = dict(_QUESTION_LIBRARY[topic])
    default = spec.get("default")
    if topic == "output_format" and prompt_flags.get("wants_notebook_hint"):
        default = ".ipynb"
    if topic == "hardware_target" and prompt_flags.get("wants_cuda_hint"):
        default = "cuda"
    if topic == "target_metric" and research_type == "quantum":
        default = "fidelity"
    if topic == "requires_quantum" and research_type == "quantum":
        default = True
    if topic == "dataset_source" and prompt_flags.get("kaggle_hint"):
        default = "kaggle"
    if topic == "framework_preference" and _normalize_topic(str(clarifications.get("algorithm_class", ""))) == "reinforcement":
        default = "pytorch"
    if topic == "quantum_qubit_count":
        parsed = _normalize_int(clarifications.get("quantum_qubit_count"))
        if parsed is not None:
            default = max(1, min(64, parsed))
    if topic == "problem_type":
        metric = _normalize_topic(str(clarifications.get("target_metric", "")))
        if metric in {"rmse", "mae", "mse", "r2"}:
            default = "regression"
    if topic == "code_level":
        if prompt_flags.get("wants_low_code_hint"):
            default = "low"
        elif prompt_flags.get("wants_advanced_code_hint"):
            default = "advanced"

    question = {
        "id": _question_id(topic, index),
        "topic": topic,
        "text": str(spec["text"]),
        "type": str(spec["type"]),
        "required": bool(spec.get("required", True)),
        "default": default,
    }
    if "options" in spec:
        question["options"] = list(spec["options"])
    return question


def _fallback_topics(
    research_type: str,
    prompt_flags: dict[str, bool],
    clarifications: dict[str, Any],
) -> list[str]:
    quantum_requested = _wants_quantum(research_type, prompt_flags, clarifications)
    dataset_source = _normalize_topic(str(clarifications.get("dataset_source", "")))
    if research_type == "quantum" or quantum_requested:
        topics = [
            "output_format",
            "quantum_framework",
            "quantum_algorithm",
            "quantum_qubit_count",
            "quantum_backend",
            "dataset_source",
            "target_metric",
            "problem_type",
            "code_level",
            "hardware_target",
            "framework_preference",
            "max_epochs",
            "random_seed",
            "python_version",
            "auto_retry_preference",
        ]
    else:
        topics = [
            "output_format",
            "algorithm_class",
            "dataset_source",
            "target_metric",
            "problem_type",
            "code_level",
            "hardware_target",
            "framework_preference",
            "max_epochs",
            "random_seed",
            "python_version",
            "auto_retry_preference",
        ]
        if prompt_flags.get("requires_quantum_hint") and "requires_quantum" not in clarifications:
            topics.insert(2, "requires_quantum")
        if quantum_requested:
            topics.insert(3, "quantum_framework")
            topics.insert(4, "quantum_algorithm")
            topics.insert(5, "quantum_qubit_count")
            topics.insert(6, "quantum_backend")
    if prompt_flags.get("kaggle_hint") or dataset_source == "kaggle":
        insert_at = min(len(topics), 4)
        topics.insert(insert_at, "kaggle_dataset_id")
    deduped: list[str] = []
    for topic in topics:
        if topic not in deduped:
            deduped.append(topic)
    return deduped


def _build_fallback_questions(
    research_type: str,
    prompt_flags: dict[str, bool],
    clarifications: dict[str, Any],
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for idx, topic in enumerate(_fallback_topics(research_type, prompt_flags, clarifications), start=1):
        questions.append(_question_from_topic(topic, idx, prompt_flags, research_type, clarifications))
        if len(questions) >= MAX_DYNAMIC_QUESTIONS:
            break
    return questions


def _extract_questions(raw: str) -> list[dict[str, Any]]:
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        return []
    if isinstance(parsed.get("questions"), list):
        return [item for item in parsed["questions"] if isinstance(item, dict)]
    parameters = parsed.get("parameters")
    if isinstance(parameters, dict) and isinstance(parameters.get("questions"), list):
        return [item for item in parameters["questions"] if isinstance(item, dict)]
    return []


def _sanitize_questions(
    llm_questions: list[dict[str, Any]],
    fallback_questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fallback_by_topic = {_normalize_topic(str(item["topic"])): item for item in fallback_questions if item.get("topic")}
    topic_order = [_normalize_topic(str(item["topic"])) for item in fallback_questions if item.get("topic")]
    sanitized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_topics: set[str] = set()

    for raw in llm_questions:
        if len(sanitized) >= MAX_DYNAMIC_QUESTIONS:
            break
        topic = _normalize_topic(str(raw.get("topic", "")))
        if not topic or topic not in _QUESTION_LIBRARY:
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        qtype = str(raw.get("type", "text")).strip().lower()
        if qtype not in _ALLOWED_TYPES:
            qtype = "text"
        qid = _question_id(topic, len(sanitized) + 1)
        if qid in seen_ids:
            continue
        item: dict[str, Any] = {
            "id": qid,
            "topic": topic,
            "text": text,
            "type": qtype,
            "required": bool(raw.get("required", True)),
            "default": raw.get("default"),
        }
        if qtype == "choice":
            options = raw.get("options")
            if isinstance(options, list):
                cleaned = [str(opt).strip() for opt in options if str(opt).strip()]
            else:
                cleaned = []
            if not cleaned and topic in fallback_by_topic:
                cleaned = list(fallback_by_topic[topic].get("options", []))
            if not cleaned:
                continue
            item["options"] = cleaned
        if item.get("default") is None and topic in fallback_by_topic:
            item["default"] = fallback_by_topic[topic].get("default")
        if topic in fallback_by_topic and item["type"] != fallback_by_topic[topic]["type"]:
            item["type"] = fallback_by_topic[topic]["type"]
            if item["type"] == "choice":
                item["options"] = list(fallback_by_topic[topic].get("options", []))
        sanitized.append(item)
        seen_ids.add(qid)
        seen_topics.add(topic)

    next_index = len(sanitized) + 1
    for topic in topic_order:
        if len(sanitized) >= MAX_DYNAMIC_QUESTIONS:
            break
        if topic in seen_topics:
            continue
        fallback = dict(fallback_by_topic[topic])
        fallback["id"] = _question_id(topic, next_index)
        next_index += 1
        sanitized.append(fallback)
        seen_topics.add(topic)

    return sanitized[:MAX_DYNAMIC_QUESTIONS]


async def _generate_question_plan(
    state: ResearchState,
    research_type: str,
    prompt_flags: dict[str, bool],
) -> list[dict[str, Any]]:
    clarifications = state.get("clarifications", {})
    fallback_questions = _build_fallback_questions(research_type, prompt_flags, clarifications)
    state_payload = {
        "research_type": research_type,
        "user_prompt": state.get("user_prompt"),
        "clarifications": clarifications,
        "prompt_flags": prompt_flags,
    }
    system_prompt = (
        SYSTEM_PROMPT.replace("{research_type}", research_type)
        .replace("{rl_policy}", "Ask only high-signal questions that are not directly inferable.")
        .replace("{state_json}", json.dumps(state_payload, indent=2, default=str))
    )
    user_prompt = (
        f"User prompt:\n{state.get('user_prompt', '')}\n\n"
        "Return one JSON object with parameters.questions."
    )
    try:
        if research_type == "quantum":
            raw = await invoke_quantum_llm_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                experiment_id=state["experiment_id"],
                phase="clarifier",
            )
        else:
            raw = await invoke_master_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                experiment_id=state["experiment_id"],
                phase="clarifier",
            )
        llm_questions = _extract_questions(raw)
        questions = _sanitize_questions(llm_questions, fallback_questions)
        if questions:
            return questions
        if not settings.ALLOW_RULE_BASED_FALLBACK:
            raise RuntimeError("Clarifier LLM returned no valid questions and rule-based fallback is disabled.")
    except Exception:
        logger.exception("agent.clarifier.llm_generation_failed", experiment_id=state["experiment_id"], research_type=research_type)
        if not settings.ALLOW_RULE_BASED_FALLBACK:
            raise
    return fallback_questions


def coerce_answer_value(question: dict[str, Any], value: Any) -> Any:
    qtype = _normalize_topic(str(question.get("type", "text")))
    if qtype == "boolean":
        normalized = _normalize_bool(value)
        if normalized is not None:
            return normalized
        default = _normalize_bool(question.get("default"))
        return default if default is not None else False
    if qtype == "number":
        normalized_number = _normalize_int(value)
        if normalized_number is not None:
            return normalized_number
        fallback_number = _normalize_int(question.get("default"))
        return fallback_number if fallback_number is not None else 0
    if qtype == "choice":
        options = [str(opt).strip() for opt in question.get("options", []) if str(opt).strip()]
        return _normalize_choice(value, options, default=question.get("default"))
    return value


async def regenerate_pending_question_state(
    state: ResearchState,
    pending: dict[str, Any],
) -> dict[str, Any] | None:
    prompt_flags = pending.get("prompt_flags", {})
    if not isinstance(prompt_flags, dict) or not prompt_flags:
        prompt_flags = _infer_prompt_flags(state.get("user_prompt", ""))
    research_type = _normalize_research_type(pending.get("research_type", state.get("research_type", "ai")))
    state["research_type"] = research_type
    if research_type == "quantum":
        state.setdefault("clarifications", {})["requires_quantum"] = True

    question_plan = await _generate_question_plan(state, research_type, prompt_flags)
    if not question_plan:
        return None

    asked_question_ids = [str(x) for x in list(pending.get("asked_question_ids") or []) if str(x).strip()]
    answered = list(pending.get("answered") or [])
    answered_topics = {_normalize_topic(str(topic)) for topic in state.get("clarifications", {}).keys()}
    max_questions = int(pending.get("max_questions") or MAX_DYNAMIC_QUESTIONS)

    next_question = None
    for item in question_plan:
        topic = _normalize_topic(str(item.get("topic", "")))
        qid = str(item.get("id", "")).strip()
        if not topic or not qid:
            continue
        if topic in answered_topics:
            continue
        if qid in asked_question_ids:
            continue
        next_question = item
        break

    if next_question is None:
        return None
    if len(asked_question_ids) >= max_questions:
        return None

    return {
        "mode": str(pending.get("mode") or "sequential_dynamic"),
        "current_question": next_question,
        "questions": [next_question],
        "question_plan": question_plan,
        "asked_question_ids": asked_question_ids,
        "answered": answered,
        "answered_count": len(answered),
        "total_questions_planned": len(question_plan),
        "max_questions": max_questions,
        "prompt_flags": prompt_flags,
        "research_type": research_type,
    }


async def clarifier_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "clarifier"
    logger.info("agent.clarifier.start", experiment_id=state["experiment_id"])
    prompt_flags = _infer_prompt_flags(state["user_prompt"])
    research_type = _normalize_research_type(state.get("research_type", "ai"))
    state["research_type"] = research_type
    if research_type == "quantum":
        state["clarifications"]["requires_quantum"] = True

    question_plan = await _generate_question_plan(state, research_type, prompt_flags)
    if not question_plan:
        raise RuntimeError("Clarifier could not generate questions")
    current = question_plan[0]

    state["pending_user_question"] = {
        "mode": "sequential_dynamic",
        "current_question": current,
        "questions": [current],  # API contract: expose active question as a single-item list.
        "question_plan": question_plan,
        "asked_question_ids": [],
        "answered": [],
        "answered_count": 0,
        "total_questions_planned": len(question_plan),
        "max_questions": MAX_DYNAMIC_QUESTIONS,
        "prompt_flags": prompt_flags,
        "research_type": research_type,
    }
    state["status"] = ExperimentStatus.WAITING.value
    state["llm_calls_count"] += 1
    logger.info(
        "agent.clarifier.end",
        experiment_id=state["experiment_id"],
        active_question_id=current.get("id"),
        research_type=research_type,
        planned_questions=len(question_plan),
    )
    return state
