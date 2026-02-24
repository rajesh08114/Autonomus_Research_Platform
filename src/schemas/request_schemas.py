from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class StartResearchRequest(BaseModel):
    prompt: str = Field(min_length=10, max_length=2000)
    priority: Literal["low", "normal", "high"] = "normal"
    tags: list[str] = Field(default_factory=list, max_length=10)
    webhook_url: HttpUrl | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class AnswerRequest(BaseModel):
    answers: dict[str, Any] = Field(min_length=1, max_length=1)


class ConfirmRequest(BaseModel):
    action_id: str
    decision: Literal["confirm", "deny"]
    reason: str = Field(default="", max_length=500)
    alternative_preference: str = ""


class AbortRequest(BaseModel):
    reason: str = "User requested cancellation"
    save_partial: bool = True


class RetryRequest(BaseModel):
    from_phase: str | None = None
    reset_retries: bool = True
    override_config: dict[str, Any] = Field(default_factory=dict)
