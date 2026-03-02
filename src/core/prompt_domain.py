from __future__ import annotations

import json
from typing import Any, Literal

from src.core.logger import get_logger
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm

logger = get_logger(__name__)

ResearchType = Literal["ai", "quantum"]
PromptDomain = Literal["ai", "quantum", "unsupported"]


def _normalize_domain(value: Any) -> PromptDomain:
    text = str(value or "").strip().lower()
    if text in {"ai", "quantum", "unsupported"}:
        return text  # type: ignore[return-value]
    return "unsupported"


async def classify_prompt_domain(
    prompt: str,
    research_type_hint: str = "ai",
    experiment_id: str | None = None,
    phase: str = "domain_classifier",
) -> dict[str, Any]:
    prompt_text = str(prompt or "").strip()
    hint = "quantum" if str(research_type_hint or "").strip().lower() == "quantum" else "ai"
    system_prompt = (
        "SYSTEM ROLE: prompt_domain_classifier.\n"
        "Classify whether the user request is an AI/ML research task, a Quantum research task, or unsupported.\n"
        "Return JSON only with keys:\n"
        "- domain: one of ['ai','quantum','unsupported']\n"
        "- reason: concise string\n"
        "- confidence: float in [0,1]\n"
        "Rules:\n"
        "- 'ai' for ML/AI/data science model development, training, evaluation, experimentation.\n"
        "- 'quantum' for quantum computing/quantum ML/circuit-focused experimentation.\n"
        "- 'unsupported' for non-AI/non-quantum domains.\n"
        "- Do not return markdown."
    )
    user_prompt = json.dumps(
        {
            "prompt": prompt_text,
            "research_type_hint": hint,
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=experiment_id,
        phase=phase,
    )
    payload = parse_json_object(raw)
    domain = _normalize_domain(payload.get("domain"))
    reason = str(payload.get("reason", "")).strip()[:500]
    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if not reason:
        reason = "Domain classification response did not include reason."
    return {"domain": domain, "reason": reason, "confidence": confidence}


async def validate_supported_prompt(
    prompt: str,
    research_type_hint: str = "ai",
    experiment_id: str | None = None,
    phase: str = "domain_classifier",
) -> tuple[bool, str, str]:
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return False, "ai", "Prompt is empty."
    classification = await classify_prompt_domain(
        prompt=prompt_text,
        research_type_hint=research_type_hint,
        experiment_id=experiment_id,
        phase=phase,
    )
    domain = _normalize_domain(classification.get("domain"))
    reason = str(classification.get("reason", "")).strip() or "No reason provided by classifier."
    if domain == "unsupported":
        return False, "ai", reason
    resolved = "quantum" if domain == "quantum" else "ai"
    return True, resolved, reason


async def resolve_research_type_from_prompt(
    prompt: str,
    requested: str = "ai",
    experiment_id: str | None = None,
    phase: str = "domain_classifier",
) -> ResearchType:
    _ = requested
    supported, resolved, reason = await validate_supported_prompt(
        prompt=prompt,
        research_type_hint=requested,
        experiment_id=experiment_id,
        phase=phase,
    )
    if not supported:
        logger.warning("prompt_domain.unsupported", reason=reason)
        return "ai"
    return "quantum" if resolved == "quantum" else "ai"
