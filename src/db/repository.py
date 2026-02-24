from __future__ import annotations

import json
import time
import uuid
from typing import Any

from src.core.logger import get_logger
from src.db.database import get_connection
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)


def _flatten_numeric_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            sub = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric_metrics(value, sub))
    elif isinstance(obj, list):
        return out
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


class ExperimentRepository:
    @staticmethod
    def _status_value(value: Any) -> str:
        return value.value if hasattr(value, "value") else str(value)

    @staticmethod
    async def create(state: ResearchState) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO experiments (
                    id, status, phase, prompt, requires_quantum, quantum_framework, framework,
                    dataset_source, hardware_target, target_metric, random_seed, retry_count,
                    llm_calls_count, total_tokens_used, project_path, documentation_path, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state["experiment_id"],
                    ExperimentRepository._status_value(state["status"]),
                    state["phase"],
                    state["user_prompt"],
                    int(state["requires_quantum"]),
                    state["quantum_framework"],
                    state["framework"],
                    state["dataset_source"],
                    state["hardware_target"],
                    state["target_metric"],
                    state["random_seed"],
                    state["retry_count"],
                    state["llm_calls_count"],
                    state["total_tokens_used"],
                    state["project_path"],
                    state["documentation_path"],
                    json.dumps(state, default=str),
                ),
            )
            conn.commit()
            logger.info("db.experiment.create", experiment_id=state["experiment_id"], phase=state["phase"], status=state["status"])
        finally:
            conn.close()

    @staticmethod
    async def update(experiment_id: str, state: ResearchState) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                UPDATE experiments
                SET status=?, phase=?, requires_quantum=?, quantum_framework=?, framework=?,
                    dataset_source=?, hardware_target=?, target_metric=?, random_seed=?,
                    retry_count=?, llm_calls_count=?, total_tokens_used=?, project_path=?,
                    documentation_path=?, state_json=?, updated_at=CURRENT_TIMESTAMP,
                    completed_at=CASE WHEN ? IN ('success','aborted','failed') THEN CURRENT_TIMESTAMP ELSE completed_at END
                WHERE id=?
                """,
                (
                    ExperimentRepository._status_value(state["status"]),
                    state["phase"],
                    int(state["requires_quantum"]),
                    state["quantum_framework"],
                    state["framework"],
                    state["dataset_source"],
                    state["hardware_target"],
                    state["target_metric"],
                    state["random_seed"],
                    state["retry_count"],
                    state["llm_calls_count"],
                    state["total_tokens_used"],
                    state["project_path"],
                    state["documentation_path"],
                    json.dumps(state, default=str),
                    ExperimentRepository._status_value(state["status"]),
                    experiment_id,
                ),
            )
            conn.commit()
            logger.info("db.experiment.update", experiment_id=experiment_id, phase=state["phase"], status=state["status"])
        finally:
            conn.close()

    @staticmethod
    async def get(experiment_id: str) -> ResearchState | None:
        conn = get_connection()
        try:
            row = conn.execute("SELECT state_json FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if not row:
                return None
            return json.loads(row["state_json"])
        finally:
            conn.close()

    @staticmethod
    async def delete(experiment_id: str) -> None:
        conn = get_connection()
        try:
            conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
            conn.execute("DELETE FROM experiment_logs WHERE experiment_id=?", (experiment_id,))
            conn.execute("DELETE FROM experiment_metrics WHERE experiment_id=?", (experiment_id,))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    async def list(
        status: str | None = None,
        phase: str | None = None,
        requires_quantum: bool | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        conn = get_connection()
        try:
            filters = []
            params: list[Any] = []
            if status:
                filters.append("status=?")
                params.append(status)
            if phase:
                filters.append("phase=?")
                params.append(phase)
            if requires_quantum is not None:
                filters.append("requires_quantum=?")
                params.append(int(requires_quantum))
            where = f"WHERE {' AND '.join(filters)}" if filters else ""
            total = conn.execute(f"SELECT COUNT(*) as c FROM experiments {where}", params).fetchone()["c"]
            rows = conn.execute(
                f"""
                SELECT id, status, phase, prompt, requires_quantum, quantum_framework, framework, target_metric,
                       llm_calls_count, total_tokens_used, retry_count, state_json, created_at, completed_at
                FROM experiments
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            return [dict(row) for row in rows], int(total)
        finally:
            conn.close()

    @staticmethod
    async def mark_failed(experiment_id: str, reason: str) -> None:
        state = await ExperimentRepository.get(experiment_id)
        if not state:
            return
        state["status"] = ExperimentStatus.FAILED.value
        state["timestamp_end"] = time.time()
        await ExperimentRepository.update(experiment_id, state)
        await ExperimentRepository.add_log(experiment_id, "system", "error", "Experiment failed", {"reason": reason})

    @staticmethod
    async def add_log(
        experiment_id: str,
        phase: str,
        level: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO experiment_logs (id, experiment_id, phase, level, message, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (f"log_{uuid.uuid4().hex[:12]}", experiment_id, phase, level, message, json.dumps(details or {})),
            )
            conn.commit()
            logger.info("db.log.add", experiment_id=experiment_id, phase=phase, level=level, message=message)
        finally:
            conn.close()

    @staticmethod
    async def get_logs(experiment_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT id, phase, level, message, details_json, timestamp
                FROM experiment_logs
                WHERE experiment_id=?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (experiment_id, limit, offset),
            ).fetchall()
            output = []
            for row in rows:
                output.append(
                    {
                        "id": row["id"],
                        "phase": row["phase"],
                        "level": row["level"],
                        "message": row["message"],
                        "details": json.loads(row["details_json"] or "{}"),
                        "timestamp": row["timestamp"],
                    }
                )
            return output
        finally:
            conn.close()

    @staticmethod
    async def add_rl_feedback(
        experiment_id: str,
        phase: str,
        reward: float,
        signal: str,
        details: dict[str, Any],
    ) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rl_feedback (
                    id              TEXT PRIMARY KEY,
                    experiment_id   TEXT,
                    phase           TEXT,
                    reward          REAL,
                    signal          TEXT,
                    details_json    TEXT,
                    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                INSERT INTO rl_feedback (id, experiment_id, phase, reward, signal, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"rl_{uuid.uuid4().hex[:12]}",
                    experiment_id,
                    phase,
                    float(reward),
                    signal,
                    json.dumps(details),
                ),
            )
            conn.commit()
            logger.info("db.rl_feedback.add", experiment_id=experiment_id, phase=phase, signal=signal, reward=reward)
        finally:
            conn.close()

    @staticmethod
    async def get_rl_phase_stats(phase: str, limit: int = 200) -> dict[str, float]:
        conn = get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rl_feedback (
                    id              TEXT PRIMARY KEY,
                    experiment_id   TEXT,
                    phase           TEXT,
                    reward          REAL,
                    signal          TEXT,
                    details_json    TEXT,
                    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            rows = conn.execute(
                """
                SELECT reward
                FROM rl_feedback
                WHERE phase=?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (phase, limit),
            ).fetchall()
            if not rows:
                return {"samples": 0.0, "avg_reward": 0.0, "positive_rate": 0.0}
            rewards = [float(row["reward"]) for row in rows]
            samples = len(rewards)
            avg_reward = sum(rewards) / samples
            positive_rate = sum(1 for r in rewards if r > 0) / samples
            return {"samples": float(samples), "avg_reward": avg_reward, "positive_rate": positive_rate}
        finally:
            conn.close()

    @staticmethod
    async def get_rl_summary(limit: int = 5000) -> dict[str, dict[str, float]]:
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT phase, signal, reward
                FROM rl_feedback
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            groups: dict[tuple[str, str], list[float]] = {}
            for row in rows:
                key = (str(row["phase"] or "unknown"), str(row["signal"] or "unknown"))
                groups.setdefault(key, []).append(float(row["reward"]))

            summary: dict[str, dict[str, float]] = {}
            for (phase, signal), rewards in groups.items():
                count = len(rewards)
                positive = sum(1 for r in rewards if r > 0)
                summary[f"{phase}:{signal}"] = {
                    "count": float(count),
                    "avg_reward": (sum(rewards) / count) if count else 0.0,
                    "positive_rate": (positive / count) if count else 0.0,
                }
            return summary
        finally:
            conn.close()

    @staticmethod
    async def add_metrics_snapshot(experiment_id: str, metrics: dict[str, Any]) -> None:
        flattened = _flatten_numeric_metrics(metrics)
        conn = get_connection()
        try:
            conn.execute("DELETE FROM experiment_metrics WHERE experiment_id=?", (experiment_id,))
            for metric_name, metric_value in flattened.items():
                conn.execute(
                    """
                    INSERT INTO experiment_metrics (id, experiment_id, metric_name, metric_value, metric_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        f"met_{uuid.uuid4().hex[:12]}",
                        experiment_id,
                        metric_name,
                        float(metric_value),
                        "numeric",
                    ),
                )
            conn.commit()
            logger.info(
                "db.metrics.snapshot",
                experiment_id=experiment_id,
                metric_count=len(flattened),
            )
        finally:
            conn.close()

    @staticmethod
    async def get_metrics_table_summary(limit: int = 50000) -> dict[str, dict[str, float]]:
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT metric_name, metric_value
                FROM experiment_metrics
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            values: dict[str, list[float]] = {}
            for row in rows:
                name = str(row["metric_name"] or "unknown")
                values.setdefault(name, []).append(float(row["metric_value"]))

            summary: dict[str, dict[str, float]] = {}
            for name, series in values.items():
                count = len(series)
                summary[name] = {
                    "count": float(count),
                    "avg": (sum(series) / count) if count else 0.0,
                    "min": min(series) if count else 0.0,
                    "max": max(series) if count else 0.0,
                }
            return summary
        finally:
            conn.close()

    @staticmethod
    async def add_llm_usage(
        provider: str,
        model: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        success: bool = True,
        error_message: str = "",
        experiment_id: str | None = None,
        phase: str | None = None,
    ) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id                  TEXT PRIMARY KEY,
                    experiment_id       TEXT,
                    phase               TEXT,
                    provider            TEXT,
                    model               TEXT,
                    latency_ms          REAL,
                    prompt_tokens       INTEGER DEFAULT 0,
                    completion_tokens   INTEGER DEFAULT 0,
                    total_tokens        INTEGER DEFAULT 0,
                    estimated_cost_usd  REAL DEFAULT 0,
                    success             BOOLEAN DEFAULT TRUE,
                    error_message       TEXT,
                    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                INSERT INTO llm_usage (
                    id, experiment_id, phase, provider, model, latency_ms,
                    prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd,
                    success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"llm_{uuid.uuid4().hex[:12]}",
                    experiment_id,
                    phase,
                    provider,
                    model,
                    float(latency_ms),
                    int(prompt_tokens),
                    int(completion_tokens),
                    int(total_tokens),
                    float(estimated_cost_usd),
                    int(bool(success)),
                    error_message[:500],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    async def get_llm_usage_summary(limit: int = 50000) -> dict[str, Any]:
        conn = get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id                  TEXT PRIMARY KEY,
                    experiment_id       TEXT,
                    phase               TEXT,
                    provider            TEXT,
                    model               TEXT,
                    latency_ms          REAL,
                    prompt_tokens       INTEGER DEFAULT 0,
                    completion_tokens   INTEGER DEFAULT 0,
                    total_tokens        INTEGER DEFAULT 0,
                    estimated_cost_usd  REAL DEFAULT 0,
                    success             BOOLEAN DEFAULT TRUE,
                    error_message       TEXT,
                    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            rows = conn.execute(
                """
                SELECT provider, model, latency_ms, prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd, success
                FROM llm_usage
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            if not rows:
                return {
                    "records": 0,
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_estimated_cost_usd": 0.0,
                    "avg_latency_ms": 0.0,
                    "provider_distribution": {},
                    "model_distribution": {},
                    "success_rate": 0.0,
                }

            provider_counts: dict[str, int] = {}
            model_counts: dict[str, int] = {}
            latencies: list[float] = []
            total_tokens = 0
            total_prompt = 0
            total_completion = 0
            total_cost = 0.0
            success = 0
            for row in rows:
                provider = str(row["provider"] or "unknown")
                model = str(row["model"] or "unknown")
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
                model_counts[model] = model_counts.get(model, 0) + 1
                latencies.append(float(row["latency_ms"] or 0.0))
                total_tokens += int(row["total_tokens"] or 0)
                total_prompt += int(row["prompt_tokens"] or 0)
                total_completion += int(row["completion_tokens"] or 0)
                total_cost += float(row["estimated_cost_usd"] or 0.0)
                success += int(row["success"] or 0)

            count = len(rows)
            return {
                "records": count,
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_estimated_cost_usd": round(total_cost, 6),
                "avg_latency_ms": (sum(latencies) / count) if count else 0.0,
                "provider_distribution": provider_counts,
                "model_distribution": model_counts,
                "success_rate": (success / count) if count else 0.0,
            }
        finally:
            conn.close()

    @staticmethod
    async def get_experiment_llm_totals(experiment_id: str) -> dict[str, float]:
        conn = get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id                  TEXT PRIMARY KEY,
                    experiment_id       TEXT,
                    phase               TEXT,
                    provider            TEXT,
                    model               TEXT,
                    latency_ms          REAL,
                    prompt_tokens       INTEGER DEFAULT 0,
                    completion_tokens   INTEGER DEFAULT 0,
                    total_tokens        INTEGER DEFAULT 0,
                    estimated_cost_usd  REAL DEFAULT 0,
                    success             BOOLEAN DEFAULT TRUE,
                    error_message       TEXT,
                    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) as total_cost
                FROM llm_usage
                WHERE experiment_id=?
                """,
                (experiment_id,),
            ).fetchone()
            return {
                "total_tokens": float(row["total_tokens"] or 0.0) if row else 0.0,
                "total_cost": float(row["total_cost"] or 0.0) if row else 0.0,
            }
        finally:
            conn.close()

    @staticmethod
    async def get_rl_reward_trend(window: int = 200) -> dict[str, float]:
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT reward
                FROM rl_feedback
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (window * 2,),
            ).fetchall()
            if not rows:
                return {"recent_avg_reward": 0.0, "previous_avg_reward": 0.0, "delta": 0.0}
            rewards = [float(row["reward"]) for row in rows]
            recent = rewards[:window]
            previous = rewards[window : window * 2]
            recent_avg = (sum(recent) / len(recent)) if recent else 0.0
            prev_avg = (sum(previous) / len(previous)) if previous else 0.0
            return {
                "recent_avg_reward": recent_avg,
                "previous_avg_reward": prev_avg,
                "delta": recent_avg - prev_avg,
            }
        finally:
            conn.close()
