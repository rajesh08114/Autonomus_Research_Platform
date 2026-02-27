from __future__ import annotations

import json
import re
from typing import Any

from src.core.logger import get_logger
from src.llm.master_llm import invoke_master_llm
from src.llm.response_parser import parse_json_response
from src.prompts.clarifier import SYSTEM_PROMPT
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

MAX_DYNAMIC_QUESTIONS = 12
_ALLOWED_TYPES = {"choice", "text", "boolean", "number"}
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disable", "disabled"}


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
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw in llm_questions:
        if len(sanitized) >= MAX_DYNAMIC_QUESTIONS:
            break
        topic = _normalize_topic(str(raw.get("topic", "")))
        if not topic:
            topic = _normalize_topic(str(raw.get("id", ""))) or f"question_{len(sanitized) + 1}"
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
            if not cleaned:
                continue
            item["options"] = cleaned
        sanitized.append(item)
        seen_ids.add(qid)

    return sanitized[:MAX_DYNAMIC_QUESTIONS]


async def _generate_question_plan(
    state: ResearchState,
    research_type: str,
    prompt_flags: dict[str, bool],
) -> list[dict[str, Any]]:
    clarifications = state.get("clarifications", {})
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
        raw = await invoke_master_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            experiment_id=state["experiment_id"],
            phase="clarifier",
        )
        llm_questions = _extract_questions(raw)
        questions = _sanitize_questions(llm_questions)
        if questions:
            return questions
        raise RuntimeError("Clarifier LLM returned no valid questions.")
    except Exception as exc:
        logger.exception("agent.clarifier.llm_generation_failed", experiment_id=state["experiment_id"], research_type=research_type)
        raise RuntimeError(f"Clarifier question generation failed: {exc}") from exc


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
        default_number = _normalize_int(question.get("default"))
        return default_number if default_number is not None else 0
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
