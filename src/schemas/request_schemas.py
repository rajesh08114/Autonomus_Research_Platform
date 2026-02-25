from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class StartResearchRequest(BaseModel):
    prompt: str = Field(min_length=10, max_length=2000)
    research_type: Literal["ai", "quantum"] = "ai"
    priority: Literal["low", "normal", "high"] = "normal"
    tags: list[str] = Field(default_factory=list, max_length=10)
    webhook_url: HttpUrl | None = None
    user_id: str | None = Field(default=None, max_length=64)
    test_mode: bool = False
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class AnswerRequest(BaseModel):
    answers: dict[str, Any] = Field(min_length=1, max_length=1)


class ConfirmRequest(BaseModel):
    action_id: str
    decision: Literal["confirm", "deny"]
    reason: str = Field(default="", max_length=500)
    alternative_preference: str = ""
    execution_result: dict[str, Any] | None = None


class AbortRequest(BaseModel):
    reason: str = "User requested cancellation"
    save_partial: bool = True


class RetryRequest(BaseModel):
    from_phase: str | None = None
    reset_retries: bool = True
    override_config: dict[str, Any] = Field(default_factory=dict)


class ChatResearchRequest(BaseModel):
    message: str = Field(min_length=3, max_length=4000)
    user_id: str | None = Field(default=None, max_length=64)
    test_mode: bool = False
    context_limit: int = Field(default=5, ge=1, le=20)


class ChatHistoryRequest(BaseModel):
    user_id: str | None = Field(default=None, max_length=64)
    test_mode: bool = False
