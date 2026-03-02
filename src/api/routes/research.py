from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from src.api.dependencies import get_request_id
from src.config.settings import settings
from src.core.logger import get_logger
from src.core.prompt_domain import validate_supported_prompt
from src.db.repository import ExperimentRepository
from src.graph.runner import (
    abort_experiment,
    get_experiment_or_404,
    retry_experiment,
    start_experiment,
    state_progress_pct,
    submit_answers,
    submit_confirmation,
    summarize_for_list,
    update_experiment_fields,
)
from src.schemas.request_schemas import AbortRequest, AnswerRequest, ConfirmRequest, RetryRequest, StartResearchRequest, UpdateResearchRequest
from src.schemas.response_schemas import error_payload, response_envelope

router = APIRouter()
logger = get_logger(__name__)


def _report_from_state_payload(state: dict[str, Any], report_path: str) -> str:
    cached = str(state.get("documentation_content") or "")
    if cached.strip():
        return cached
    plan_items = state.get("local_file_plan", [])
    if not report_path or not isinstance(plan_items, list):
        return ""
    for item in reversed(plan_items):
        if not isinstance(item, dict):
            continue
        if str(item.get("path", "")) != report_path:
            continue
        content = str(item.get("content", ""))
        if content.strip():
            return content
    return ""


@router.post("/research/start", status_code=201)
async def post_start(request: StartResearchRequest, request_id: str = Depends(get_request_id)):
    logger.info("api.research.start", request_id=request_id, prompt_len=len(request.prompt), research_type=request.research_type)
    if settings.effective_master_llm_provider == "huggingface" and not settings.huggingface_api_key:
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                "LLM_CONFIGURATION_ERROR",
                "HF_API_KEY (or MASTER_LLM_API_KEY) is required.",
            ),
        )
    try:
        supported, resolved_research_type, reason = await validate_supported_prompt(
            request.prompt,
            research_type_hint=request.research_type,
            phase="api_start_domain_gate",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=error_payload(
                "DOMAIN_CLASSIFIER_UNAVAILABLE",
                "Domain classification service is unavailable. Try again.",
                details={"reason": str(exc)},
            ),
        ) from exc
    if not supported:
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                "UNSUPPORTED_RESEARCH_DOMAIN",
                "Only AI and Quantum research prompts are supported.",
                details={"reason": reason},
            ),
        )
    experiment_id = await start_experiment(
        request.prompt,
        request.config_overrides,
        research_type=resolved_research_type,
        user_id=request.user_id,
        test_mode=request.test_mode,
        webhook_url=str(request.webhook_url) if request.webhook_url else None,
    )
    state = await get_experiment_or_404(experiment_id)
    data = {
        "experiment_id": experiment_id,
        "status": state["status"],
        "phase": state["phase"],
        "research_type": state.get("research_type", resolved_research_type),
        "created_at": state["timestamp_start"],
        "execution_target": state.get("execution_target", "local_machine"),
        "execution_mode": state.get("execution_mode", "vscode_extension"),
        "default_allow_research": bool(state.get("default_allow_research", False)),
        "llm": {"provider": state.get("llm_provider"), "model": state.get("llm_model")},
        "research_scope": {
            "user_id": state.get("research_user_id", "anonymous"),
            "test_mode": bool(state.get("test_mode", False)),
            "collection_key": state.get("collection_key"),
            "webhook_enabled": bool(state.get("webhook_url")),
        },
        "estimated_duration_minutes": 15,
        "pending_questions": state.get("pending_user_question"),
        "links": {
            "self": f"/api/v1/research/{experiment_id}",
            "answer": f"/api/v1/research/{experiment_id}/answer",
            "status": f"/api/v1/research/{experiment_id}/status",
            "logs": f"/api/v1/research/{experiment_id}/logs",
        },
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.post("/research/{experiment_id}/answer")
async def post_answer(experiment_id: str, request: AnswerRequest, request_id: str = Depends(get_request_id)):
    logger.info("api.research.answer", request_id=request_id, experiment_id=experiment_id, answer_count=len(request.answers))
    try:
        state = await submit_answers(experiment_id, request.answers)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=error_payload("WRONG_STATUS", str(exc)))

    data = {
        "experiment_id": experiment_id,
        "research_type": state.get("research_type", "ai"),
        "answers_received": len(request.answers),
        "answered_question_ids": list(request.answers.keys()),
        "status": state["status"],
        "phase": state["phase"],
        "pending_questions": state.get("pending_user_question"),
        "question_progress": {
            "answered_count": ((state.get("pending_user_question") or {}).get("answered_count", 0)),
            "total_planned": ((state.get("pending_user_question") or {}).get("total_questions_planned", 0)),
        },
        "message": "Answer accepted. Next question queued." if state["status"] == "waiting_user" else "Clarification complete. Workflow advanced.",
        "next_action": "answer_next_question" if state["status"] == "waiting_user" else "wait",
        "estimated_next_update_seconds": 30,
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.post("/research/{experiment_id}/confirm")
async def post_confirm(experiment_id: str, request: ConfirmRequest, request_id: str = Depends(get_request_id)):
    logger.info("api.research.confirm", request_id=request_id, experiment_id=experiment_id, decision=request.decision)
    try:
        state = await submit_confirmation(
            experiment_id=experiment_id,
            action_id=request.action_id,
            decision=request.decision,
            reason=request.reason,
            alternative_preference=request.alternative_preference,
            execution_result=request.execution_result,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=error_payload("WRONG_STATUS", str(exc)))

    data = {
        "experiment_id": experiment_id,
        "research_type": state.get("research_type", "ai"),
        "action_id": request.action_id,
        "decision": request.decision,
        "status": state["status"],
        "phase": state["phase"],
        "pending_action": state.get("pending_user_confirm"),
        "message": "Confirmation processed.",
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.patch("/research/{experiment_id}")
async def patch_research(experiment_id: str, request: UpdateResearchRequest, request_id: str = Depends(get_request_id)):
    logger.info(
        "api.research.patch",
        request_id=request_id,
        experiment_id=experiment_id,
        update_keys=sorted((request.updates or {}).keys()),
        merge_nested=request.merge_nested,
    )
    try:
        state, applied, rejected = await update_experiment_fields(
            experiment_id=experiment_id,
            updates=request.updates,
            merge_nested=bool(request.merge_nested),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=error_payload("WRONG_STATUS", str(exc)))

    if not applied:
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                "INVALID_UPDATE_FIELDS",
                "No valid updates were applied.",
                details={"rejected_fields": rejected},
            ),
        )

    data = {
        "experiment_id": experiment_id,
        "status": state["status"],
        "phase": state["phase"],
        "research_type": state.get("research_type", "ai"),
        "updated_fields": sorted(applied.keys()),
        "applied_updates": applied,
        "rejected_fields": rejected,
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/research/{experiment_id}")
async def get_research(experiment_id: str, request_id: str = Depends(get_request_id)):
    logger.info("api.research.get", request_id=request_id, experiment_id=experiment_id)
    try:
        state = await get_experiment_or_404(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))

    data = {
        "experiment_id": state["experiment_id"],
        "status": state["status"],
        "phase": state["phase"],
        "created_at": state["timestamp_start"],
        "updated_at": state["timestamp_end"] or state["timestamp_start"],
        "prompt": state["user_prompt"],
        "research_type": state.get("research_type", "ai"),
        "requires_quantum": state["requires_quantum"],
        "quantum_framework": state["quantum_framework"],
        "framework": state["framework"],
        "dataset_source": state["dataset_source"],
        "target_metric": state["target_metric"],
        "hardware_target": state["hardware_target"],
        "retry_count": state["retry_count"],
        "llm_calls_count": state.get("llm_calls_count", 0),
        "total_tokens_used": state.get("total_tokens_used", 0),
        "confirmations_requested": state.get("confirmations_requested", 0),
        "confirmations_processed": state.get("confirmations_processed", 0),
        "phase_timings": state["phase_timings"],
        "created_files": state["created_files"],
        "installed_packages": state["installed_packages"],
        "denied_actions": state["denied_actions"],
        "errors": state["errors"],
        "metrics": state["metrics"],
        "llm_total_cost_usd": state.get("llm_total_cost_usd", 0.0),
        "execution_target": state.get("execution_target", "local_machine"),
        "execution_mode": state.get("execution_mode", "vscode_extension"),
        "default_allow_research": bool(state.get("default_allow_research", False)),
        "llm": {"provider": state.get("llm_provider"), "model": state.get("llm_model")},
        "pending_questions": state.get("pending_user_question"),
        "pending_action": state.get("pending_user_confirm"),
        "research_scope": {
            "user_id": state.get("research_user_id", "anonymous"),
            "test_mode": bool(state.get("test_mode", False)),
            "collection_key": state.get("collection_key"),
        },
        "links": {
            "logs": f"/api/v1/research/{experiment_id}/logs",
            "files": f"/api/v1/research/{experiment_id}/files",
            "results": f"/api/v1/research/{experiment_id}/results",
            "report": f"/api/v1/research/{experiment_id}/report",
            "abort": f"/api/v1/research/{experiment_id}/abort",
        },
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/research/{experiment_id}/status")
async def get_status(
    experiment_id: str,
    include_phase_timings: bool = Query(default=False),
    include_last_action: bool = Query(default=False),
    request_id: str = Depends(get_request_id),
):
    logger.info("api.research.status", request_id=request_id, experiment_id=experiment_id)
    try:
        state = await get_experiment_or_404(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))

    data = {
        "experiment_id": experiment_id,
        "status": state["status"],
        "phase": state["phase"],
        "research_type": state.get("research_type", "ai"),
        "retry_count": state["retry_count"],
        "llm_calls_count": state.get("llm_calls_count", 0),
        "confirmations_requested": state.get("confirmations_requested", 0),
        "confirmations_processed": state.get("confirmations_processed", 0),
        "current_script": state["current_script"],
        "progress_pct": state_progress_pct(state),
        "waiting_for_user": state["status"] == "waiting_user",
        "pending_questions": state.get("pending_user_question"),
        "pending_action": state.get("pending_user_confirm"),
        "execution_target": state.get("execution_target", "local_machine"),
        "execution_mode": state.get("execution_mode", "vscode_extension"),
        "default_allow_research": bool(state.get("default_allow_research", False)),
        "llm_provider": state.get("llm_provider"),
        "llm_model": state.get("llm_model"),
        "last_updated": state.get("timestamp_end") or state["timestamp_start"],
        "elapsed_seconds": int((state.get("timestamp_end") or time.time()) - state["timestamp_start"]),
    }
    if include_phase_timings:
        data["phase_timings"] = state.get("phase_timings", {})
    if include_last_action:
        data["last_action"] = state.get("execution_logs", [])[-1] if state.get("execution_logs") else None
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/research/{experiment_id}/logs")
async def get_logs(
    experiment_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    request_id: str = Depends(get_request_id),
):
    logger.info("api.research.logs", request_id=request_id, experiment_id=experiment_id, limit=limit, offset=offset)
    try:
        state = await get_experiment_or_404(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))

    logs = await ExperimentRepository.get_logs(experiment_id, limit=limit, offset=offset)
    data = {
        "experiment_id": experiment_id,
        "total_entries": len(logs),
        "returned": len(logs),
        "logs": logs,
        "execution_logs": state.get("execution_logs", []),
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/research/{experiment_id}/results")
async def get_results(experiment_id: str, request_id: str = Depends(get_request_id)):
    logger.info("api.research.results", request_id=request_id, experiment_id=experiment_id)
    try:
        state = await get_experiment_or_404(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))

    if state["status"] not in {"success", "aborted", "failed"}:
        return response_envelope(
            True,
            data={
                "experiment_id": experiment_id,
                "status": state["status"],
                "phase": state["phase"],
                "message": "Experiment still in progress. Results not yet available.",
                "progress_pct": state_progress_pct(state),
            },
            request_id=request_id,
        )

    return response_envelope(True, data=state.get("metrics", {}), request_id=request_id)


@router.get("/research/{experiment_id}/report")
async def get_report(
    experiment_id: str,
    format: str = Query(default="markdown", pattern="^(markdown|json)$"),
    download: bool = Query(default=False),
    request_id: str = Depends(get_request_id),
):
    logger.info("api.research.report", request_id=request_id, experiment_id=experiment_id, format=format, download=download)
    try:
        state = await get_experiment_or_404(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))

    report = str(state.get("documentation_path") or "")
    content = ""
    if report and Path(report).exists():
        content = Path(report).read_text(encoding="utf-8")
    if not content:
        content = _report_from_state_payload(state, report)
    if not content:
        raise HTTPException(status_code=404, detail=error_payload("REPORT_NOT_FOUND", "Report not available yet"))
    report_path_value = report or f"{state['project_path']}/docs/final_report.md"
    if download:
        return PlainTextResponse(content, headers={"Content-Disposition": f"attachment; filename={Path(report_path_value).name}"})
    data = {
        "experiment_id": experiment_id,
        "report_path": report_path_value,
        "generated_at": state.get("timestamp_end"),
        "word_count": len(content.split()),
        "sections": state.get("report_sections", []),
        "content": content if format == "markdown" else {"markdown": content},
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/research")
async def list_research(
    status: str | None = None,
    phase: str | None = None,
    requires_quantum: bool | None = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    request_id: str = Depends(get_request_id),
):
    logger.info("api.research.list", request_id=request_id, limit=limit, offset=offset, status=status, phase=phase)
    rows, total = await ExperimentRepository.list(
        status=status,
        phase=phase,
        requires_quantum=requires_quantum,
        limit=limit,
        offset=offset,
    )
    data = {
        "experiments": [summarize_for_list(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "next": f"/api/v1/research?offset={offset + limit}" if offset + limit < total else None,
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.delete("/research/{experiment_id}/abort")
async def delete_abort(experiment_id: str, request: AbortRequest, request_id: str = Depends(get_request_id)):
    logger.warning("api.research.abort", request_id=request_id, experiment_id=experiment_id, reason=request.reason)
    try:
        state = await abort_experiment(experiment_id, request.reason, request.save_partial)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))
    data = {
        "experiment_id": experiment_id,
        "status": state["status"],
        "aborted_at": state["timestamp_end"],
        "aborted_phase": state["phase"],
        "partial_saved": request.save_partial,
        "partial_path": f"{state['project_path']}/outputs/partial/",
        "error_report": f"{state['project_path']}/docs/error_report.md",
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.post("/research/{experiment_id}/retry", status_code=202)
async def post_retry(experiment_id: str, request: RetryRequest, request_id: str = Depends(get_request_id)):
    logger.info("api.research.retry", request_id=request_id, experiment_id=experiment_id, from_phase=request.from_phase)
    try:
        state = await retry_experiment(
            experiment_id=experiment_id,
            from_phase=request.from_phase,
            reset_retries=request.reset_retries,
            override_config=request.override_config,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))
    data = {
        "experiment_id": experiment_id,
        "status": state["status"],
        "restarted_from": request.from_phase or "error_recovery",
        "retry_count": state["retry_count"],
        "message": "Experiment resumed from checkpoint",
    }
    return response_envelope(True, data=data, request_id=request_id)
