from __future__ import annotations

from collections import Counter, defaultdict
import json
import statistics
import shutil

from fastapi import APIRouter, Depends

from src.api.dependencies import get_request_id
from src.config.settings import settings
from src.core.logger import get_logger
from src.db.database import get_connection
from src.db.repository import ExperimentRepository
from src.schemas.response_schemas import response_envelope

router = APIRouter()
logger = get_logger(__name__)


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _latency_bin(seconds: float) -> str:
    if seconds < 0.5:
        return "<0.5s"
    if seconds < 2:
        return "0.5-2s"
    if seconds < 5:
        return "2-5s"
    if seconds < 30:
        return "5-30s"
    return ">=30s"


@router.get("/system/health")
async def get_health(request_id: str = Depends(get_request_id)):
    logger.info("api.system.health", request_id=request_id)
    db_status = "up"
    db_size_mb = 0.0
    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        db_path = settings.state_db_path
        if db_path.exists():
            db_size_mb = round(db_path.stat().st_size / 1e6, 2)
    except Exception:
        db_status = "down"

    project_root = settings.project_root_path
    project_root.mkdir(parents=True, exist_ok=True)
    free_gb = round(shutil.disk_usage(project_root).free / (1024**3), 2)

    running_rows, running_total = await ExperimentRepository.list(status="running", limit=1, offset=0)
    all_rows, total = await ExperimentRepository.list(limit=1, offset=0)
    _ = (running_rows, all_rows)

    data = {
        "status": "healthy" if db_status == "up" else "degraded",
        "version": "2.0.0",
        "components": {
            "api": {"status": "up", "latency_ms": 1},
            "langgraph": {"status": "up"},
            "master_llm": {
                "status": "up" if settings.MASTER_LLM_PROVIDER == "rule_based" or settings.huggingface_api_key else "down",
                "provider": settings.MASTER_LLM_PROVIDER,
                "model": settings.huggingface_model_id
                if settings.MASTER_LLM_PROVIDER.strip().lower() in {"huggingface", "hf", "hugging_face"}
                else settings.MASTER_LLM_MODEL,
            },
            "quantum_llm": {"status": "up" if settings.QUANTUM_LLM_ENDPOINT else "degraded", "endpoint": settings.QUANTUM_LLM_ENDPOINT},
            "database": {"status": db_status, "type": "sqlite", "size_mb": db_size_mb},
            "filesystem": {"status": "up", "free_gb": free_gb},
        },
        "active_experiments": running_total,
        "total_experiments": total,
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/system/metrics")
async def get_metrics(request_id: str = Depends(get_request_id)):
    logger.info("api.system.metrics", request_id=request_id)
    rows, total = await ExperimentRepository.list(limit=10000, offset=0)
    counts = {"success": 0, "failed": 0, "aborted": 0, "running": 0, "pending": 0, "waiting_user": 0}
    durations: list[float] = []
    retry_values: list[int] = []
    healed_candidates = 0
    healed_success = 0
    llm_calls = 0
    llm_tokens = 0
    quantum_total = 0
    quantum_frameworks = {"pennylane": 0, "qiskit": 0, "cirq": 0}
    llm_provider_counts: dict[str, int] = {}
    llm_model_counts: dict[str, int] = {}
    phase_latency_histograms: dict[str, Counter[str]] = defaultdict(Counter)
    error_clusters: Counter[str] = Counter()

    for row in rows:
        status = row["status"]
        if status in counts:
            counts[status] += 1
        llm_calls += int(row.get("llm_calls_count") or 0)
        llm_tokens += int(row.get("total_tokens_used") or 0)
        retry = int(row.get("retry_count") or 0)
        retry_values.append(retry)

        if bool(row.get("requires_quantum")):
            quantum_total += 1
            framework = str(row.get("quantum_framework") or "").lower()
            if framework in quantum_frameworks:
                quantum_frameworks[framework] += 1

        state_json = row.get("state_json")
        if state_json:
            try:
                state = json.loads(state_json)
            except Exception:
                state = {}
        else:
            state = {}

        provider = str(state.get("llm_provider") or settings.MASTER_LLM_PROVIDER)
        model = str(state.get("llm_model") or settings.MASTER_LLM_MODEL)
        llm_provider_counts[provider] = llm_provider_counts.get(provider, 0) + 1
        llm_model_counts[model] = llm_model_counts.get(model, 0) + 1

        ts_start = _as_float(state.get("timestamp_start"))
        ts_end = _as_float(state.get("timestamp_end"))
        total_duration = _as_float(state.get("total_duration_sec"))
        if total_duration is not None:
            durations.append(total_duration)
        elif ts_start is not None and ts_end is not None and ts_end >= ts_start:
            durations.append(ts_end - ts_start)

        phase_timings = state.get("phase_timings", {}) if isinstance(state.get("phase_timings"), dict) else {}
        for phase, duration in phase_timings.items():
            d = _as_float(duration)
            if d is not None:
                phase_latency_histograms[str(phase)][_latency_bin(float(d))] += 1

        errors = state.get("errors", []) if isinstance(state.get("errors"), list) else []
        for item in errors:
            if isinstance(item, dict):
                category = str(item.get("category") or "unknown")
                error_clusters[category] += 1

        if retry > 0:
            healed_candidates += 1
            if status == "success":
                healed_success += 1

    success = counts["success"]
    failed = counts["failed"]
    aborted = counts["aborted"]
    running = counts["running"]
    success_rate = float(success / total) if total else 0.0
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0
    median_duration = statistics.median(durations) if durations else 0.0
    avg_retry = (sum(retry_values) / len(retry_values)) if retry_values else 0.0
    self_heal_rate = (healed_success / healed_candidates) if healed_candidates else 0.0
    rl_summary = await ExperimentRepository.get_rl_summary(limit=5000)
    rl_trend = await ExperimentRepository.get_rl_reward_trend(window=200)
    llm_usage = await ExperimentRepository.get_llm_usage_summary(limit=50000)
    metrics_table_summary = await ExperimentRepository.get_metrics_table_summary(limit=50000)
    top_provider = max(llm_provider_counts.items(), key=lambda item: item[1])[0] if llm_provider_counts else settings.MASTER_LLM_PROVIDER
    top_model = max(llm_model_counts.items(), key=lambda item: item[1])[0] if llm_model_counts else settings.MASTER_LLM_MODEL

    conn = get_connection()
    try:
        log_rows = conn.execute(
            """
            SELECT phase, level, message
            FROM experiment_logs
            ORDER BY timestamp DESC
            LIMIT 200000
            """
        ).fetchall()
    finally:
        conn.close()

    phase_started: Counter[str] = Counter()
    phase_completed: Counter[str] = Counter()
    phase_errors: Counter[str] = Counter()
    for row in log_rows:
        phase = str(row["phase"] or "unknown")
        level = str(row["level"] or "unknown").lower()
        message = str(row["message"] or "")
        if message.startswith("Phase ") and message.endswith(" started"):
            parts = message.split()
            if len(parts) >= 2:
                phase_started[parts[1]] += 1
            else:
                phase_started[phase] += 1
        if message.startswith("Phase ") and message.endswith(" completed"):
            parts = message.split()
            if len(parts) >= 2:
                phase_completed[parts[1]] += 1
            else:
                phase_completed[phase] += 1
        if level == "error":
            phase_errors[phase] += 1

    agent_success_ratios: dict[str, dict[str, float]] = {}
    all_phases = set(phase_started) | set(phase_completed) | set(phase_errors)
    for phase in sorted(all_phases):
        started = int(phase_started.get(phase, 0))
        completed = int(phase_completed.get(phase, 0))
        errors = int(phase_errors.get(phase, 0))
        denom = max(1, started, completed + errors)
        agent_success_ratios[phase] = {
            "started": float(started),
            "completed": float(completed),
            "errors": float(errors),
            "completion_ratio": min(1.0, completed / denom),
            "mismatch_events": float(max(0, completed - started)),
        }

    data = {
        "experiments": {
            "total": total,
            "success": success,
            "failed": failed,
            "aborted": aborted,
            "running": running,
            "success_rate": round(success_rate, 3),
        },
        "performance": {
            "avg_duration_sec": round(float(avg_duration), 4),
            "median_duration_sec": round(float(median_duration), 4),
            "avg_retry_count": round(float(avg_retry), 4),
            "error_self_heal_rate": round(float(self_heal_rate), 4),
        },
        "llm_usage": {
            "total_calls": int(llm_usage.get("records", llm_calls)),
            "total_tokens": int(llm_usage.get("total_tokens", llm_tokens)),
            "total_prompt_tokens": int(llm_usage.get("total_prompt_tokens", 0)),
            "total_completion_tokens": int(llm_usage.get("total_completion_tokens", 0)),
            "total_estimated_cost_usd": float(llm_usage.get("total_estimated_cost_usd", 0.0)),
            "avg_latency_ms": round(float(llm_usage.get("avg_latency_ms", 0.0)), 4),
            "success_rate": round(float(llm_usage.get("success_rate", 0.0)), 4),
            "avg_calls_per_experiment": round(float(llm_calls / total), 4) if total else 0.0,
            "provider_distribution": llm_usage.get("provider_distribution", llm_provider_counts),
            "model_distribution": llm_usage.get("model_distribution", llm_model_counts),
            "active_provider": top_provider,
            "active_model": top_model,
        },
        "quantum_experiments": {"total": quantum_total, "frameworks": quantum_frameworks},
        "rl_feedback": {
            "records": int(sum(int(v["count"]) for v in rl_summary.values())),
            "by_phase_signal": rl_summary,
            "reward_trend": rl_trend,
        },
        "observability": {
            "phase_latency_histograms": {phase: dict(counter) for phase, counter in phase_latency_histograms.items()},
            "agent_success_ratios": agent_success_ratios,
            "failure_clusters": dict(error_clusters),
            "metrics_table_summary": metrics_table_summary,
        },
        "execution": {"target": "local_machine"},
    }
    return response_envelope(True, data=data, request_id=request_id)
