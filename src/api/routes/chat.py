from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_request_id
from src.config.settings import settings
from src.core.chat_assistant import generate_chat_response, select_relevant_history, state_to_summary
from src.core.history_scope import build_collection_key, normalize_user_id
from src.core.logger import get_logger
from src.db.repository import ExperimentRepository
from src.schemas.request_schemas import ChatResearchRequest
from src.schemas.response_schemas import error_payload, response_envelope

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat/research")
async def post_chat_research(request: ChatResearchRequest, request_id: str = Depends(get_request_id)):
    if (
        settings.effective_master_llm_provider == "huggingface"
        and not settings.huggingface_api_key
    ):
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                "LLM_CONFIGURATION_ERROR",
                "HF_API_KEY (or MASTER_LLM_API_KEY) is required.",
            ),
        )

    user_id = normalize_user_id(request.user_id)
    collection_key = build_collection_key(user_id, request.test_mode)
    context_limit = max(1, min(int(request.context_limit), int(settings.CHAT_CONTEXT_LIMIT_MAX)))
    lookup_limit = max(context_limit * 3, int(settings.CHAT_CONTEXT_LIMIT_DEFAULT) * 3)

    logger.info(
        "api.chat.research",
        request_id=request_id,
        collection_key=collection_key,
        user_id=user_id,
        test_mode=request.test_mode,
        context_limit=context_limit,
    )

    history_states = await ExperimentRepository.get_collection_states(collection_key, limit=lookup_limit)
    history_source = "collection"
    if request.test_mode and not history_states:
        history_states = await ExperimentRepository.get_recent_states(limit=lookup_limit)
        history_source = "unified_db_global"

    history_items = [state_to_summary(state) for state in history_states]
    selected = select_relevant_history(request.message, history_items, limit=context_limit)
    chat_history = await ExperimentRepository.get_chat_history(collection_key, limit=int(settings.CHAT_HISTORY_LIMIT))

    await ExperimentRepository.add_chat_message(
        collection_key=collection_key,
        user_id=user_id,
        test_mode=request.test_mode,
        role="user",
        message=request.message,
        metadata={"request_id": request_id},
    )

    generated = await generate_chat_response(
        question=request.message,
        selected_history=selected,
        chat_history=chat_history,
        test_mode=request.test_mode,
    )

    assistant_metadata = {
        "request_id": request_id,
        "history_source": history_source,
        "references": [item.get("experiment_id") for item in selected if item.get("experiment_id")],
        "generation": generated.get("generation", {}),
        "token_usage": generated.get("token_usage", {}),
    }
    await ExperimentRepository.add_chat_message(
        collection_key=collection_key,
        user_id=user_id,
        test_mode=request.test_mode,
        role="assistant",
        message=str(generated.get("answer") or ""),
        metadata=assistant_metadata,
    )

    data = {
        "answer": generated.get("answer"),
        "follow_up_questions": generated.get("follow_up_questions", []),
        "scope": {
            "user_id": user_id,
            "test_mode": bool(request.test_mode),
            "collection_key": collection_key,
        },
        "retrieval": {
            "history_source": history_source,
            "history_loaded": len(history_items),
            "history_used": len(selected),
            "references": selected,
        },
        "generation": generated.get("generation", {}),
        "token_usage": generated.get("token_usage", {}),
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/chat/history")
async def get_chat_history(
    user_id: str | None = None,
    test_mode: bool = Query(default=False),
    limit: int = Query(default=40, ge=1, le=200),
    request_id: str = Depends(get_request_id),
):
    normalized_user = normalize_user_id(user_id)
    collection_key = build_collection_key(normalized_user, test_mode)
    logger.info(
        "api.chat.history",
        request_id=request_id,
        collection_key=collection_key,
        user_id=normalized_user,
        test_mode=test_mode,
        limit=limit,
    )
    messages = await ExperimentRepository.get_chat_history(collection_key, limit=min(limit, int(settings.CHAT_HISTORY_LIMIT)))
    data = {
        "scope": {
            "user_id": normalized_user,
            "test_mode": bool(test_mode),
            "collection_key": collection_key,
        },
        "messages": messages,
        "count": len(messages),
    }
    return response_envelope(True, data=data, request_id=request_id)
