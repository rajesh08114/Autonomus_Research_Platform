from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.logger import get_logger
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ResearchState

logger = get_logger(__name__)


def _default_execution_order(state: ResearchState, project: Path) -> list[str]:
    code_level = str((state.get("research_plan") or {}).get("code_level", "intermediate")).strip().lower()
    algorithm_class = str((state.get("research_plan") or {}).get("algorithm_class", "supervised")).strip().lower()
    order: list[str] = []
    if code_level == "advanced":
        order.append(str(project / "data" / "validate_data.py"))
    if algorithm_class in {"reinforcement", "quantum_ml"}:
        order.append(str(project / "main.py"))
    else:
        order.extend([str(project / "main.py")])
    return order


async def _invoke_dynamic_scheduler(state: ResearchState, candidates: list[str]) -> dict[str, Any]:
    system_prompt = (
        "SYSTEM ROLE: job_scheduler_dynamic_plan.\n"
        "Return JSON only with keys:\n"
        "- execution_order: array of absolute script paths\n"
        "- rationale: short string\n"
        "Use only candidate scripts from context."
    )
    user_prompt = json.dumps(
        {
            "candidates": candidates,
            "research_plan": state.get("research_plan", {}),
            "requires_quantum": state.get("requires_quantum"),
            "framework": state.get("framework"),
            "dataset_source": state.get("dataset_source"),
            "current_phase": state.get("phase"),
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="job_scheduler",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = parse_json_object(raw)
    if not parsed:
        logger.warning("agent.job_scheduler.dynamic_parse_failed", experiment_id=state["experiment_id"])
    return parsed


def _sanitize_dynamic_execution_order(
    *,
    payload: dict[str, Any],
    candidates: list[str],
    project: Path,
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    candidate_set = {str(Path(item).resolve()) for item in candidates}
    raw_order = payload.get("execution_order")
    if not isinstance(raw_order, list):
        return [], ["execution_order must be an array"]
    order: list[str] = []
    for idx, item in enumerate(raw_order):
        path_text = str(item or "").strip()
        if not path_text:
            violations.append(f"execution_order[{idx}] empty path")
            continue
        resolved = str(Path(path_text).resolve())
        if resolved not in candidate_set:
            violations.append(f"execution_order[{idx}] not in candidates: {resolved}")
            continue
        try:
            Path(resolved).resolve().relative_to(project.resolve())
        except ValueError:
            violations.append(f"execution_order[{idx}] outside project: {resolved}")
            continue
        if resolved not in order:
            order.append(resolved)
    return order, violations


async def job_scheduler_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "job_scheduler"
    logger.info("agent.scheduler.start", experiment_id=state["experiment_id"])
    project = Path(state["project_path"])
    default_order = _default_execution_order(state, project)
    candidates = []
    for script in default_order:
        resolved = str(Path(script).resolve())
        if Path(resolved).exists():
            candidates.append(resolved)

    used_dynamic = False
    fallback_static = False
    execution_order = list(candidates)
    if candidates:
        payload = await _invoke_dynamic_scheduler(state, candidates)
        if not payload:
            if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                fallback_static = True
                logger.warning("agent.job_scheduler.dynamic_fallback_static", experiment_id=state["experiment_id"], reason="parse_failed")
            else:
                raise RuntimeError("Job scheduler dynamic parse failed")
        else:
            dynamic_order, violations = _sanitize_dynamic_execution_order(payload=payload, candidates=candidates, project=project)
            if violations:
                logger.warning("agent.job_scheduler.dynamic_validation_failed", experiment_id=state["experiment_id"], violations=violations)
                if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                    fallback_static = True
                    logger.warning(
                        "agent.job_scheduler.dynamic_fallback_static",
                        experiment_id=state["experiment_id"],
                        reason="validation_failed",
                    )
                else:
                    raise RuntimeError(f"Job scheduler dynamic validation failed: {violations}")
            else:
                execution_order = dynamic_order
                used_dynamic = True

    main_path = str((project / "main.py").resolve())
    if main_path in candidates and main_path not in execution_order:
        execution_order.append(main_path)

    state["execution_order"] = execution_order
    state.setdefault("research_plan", {})["scheduler_dynamic_plan_summary"] = {
        "used_dynamic": used_dynamic,
        "fallback_static": fallback_static,
        "candidate_count": len(candidates),
        "execution_order_count": len(execution_order),
    }
    logger.info("agent.scheduler.end", experiment_id=state["experiment_id"], execution_order=state["execution_order"])
    return state

