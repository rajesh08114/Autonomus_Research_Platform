from __future__ import annotations

import logging
from typing import Any

try:
    import structlog as _structlog
except Exception:  # pragma: no cover - fallback path when structlog is missing
    _structlog = None


class _FallbackLogger:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _emit(self, log_level: str, event: str, **kwargs: Any) -> None:
        message = event
        if kwargs:
            kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{event} | {kv}"
        getattr(self._logger, log_level)(message)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit("debug", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit("error", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._emit("exception", event, **kwargs)


def has_structlog() -> bool:
    return _structlog is not None


def get_logger(name: str | None = None):
    if _structlog is not None:
        return _structlog.get_logger(name)
    return _FallbackLogger(logging.getLogger(name or "app"))


def configure_structlog() -> None:
    if _structlog is None:
        return
    _structlog.configure(
        processors=[
            _structlog.contextvars.merge_contextvars,
            _structlog.stdlib.add_log_level,
            _structlog.processors.TimeStamper(fmt="iso"),
            _structlog.processors.StackInfoRenderer(),
            _structlog.processors.format_exc_info,
            _structlog.processors.JSONRenderer(),
        ],
        wrapper_class=_structlog.stdlib.BoundLogger,
        logger_factory=_structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
