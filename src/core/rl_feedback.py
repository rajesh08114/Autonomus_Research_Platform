from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import settings
from src.db.repository import ExperimentRepository


@dataclass(slots=True)
class RLPhaseStats:
    phase: str
    samples: int
    avg_reward: float
    positive_rate: float


async def record_phase_feedback(
    experiment_id: str,
    phase: str,
    reward: float,
    signal: str,
    details: dict | None = None,
) -> None:
    if not settings.RL_ENABLED:
        return
    await ExperimentRepository.add_rl_feedback(
        experiment_id=experiment_id,
        phase=phase,
        reward=reward,
        signal=signal,
        details=details or {},
    )


async def get_phase_stats(phase: str) -> RLPhaseStats:
    if not settings.RL_ENABLED:
        return RLPhaseStats(phase=phase, samples=0, avg_reward=0.0, positive_rate=0.0)
    stats = await ExperimentRepository.get_rl_phase_stats(phase=phase, limit=settings.RL_FEEDBACK_WINDOW)
    return RLPhaseStats(
        phase=phase,
        samples=int(stats["samples"]),
        avg_reward=float(stats["avg_reward"]),
        positive_rate=float(stats["positive_rate"]),
    )


async def get_phase_policy_hint(phase: str) -> str:
    stats = await get_phase_stats(phase)
    if stats.samples < settings.RL_MIN_SAMPLES_FOR_POLICY:
        return (
            "Policy: prioritize correctness first, keep outputs minimal, "
            "and never bypass schema or safety checks."
        )

    if stats.positive_rate < 0.4:
        return (
            "Policy: low positive reward rate detected. Slow down decisions, "
            "explicitly validate assumptions, and prefer safe fallback strategies."
        )
    if stats.avg_reward < 0.2:
        return (
            "Policy: high-risk phase detected from historical rewards. "
            "Use conservative decisions, strict schema adherence, and explicit fallback handling."
        )
    if stats.avg_reward < 0.6:
        return (
            "Policy: moderate performance. Improve validation rigor, reduce ambiguity, "
            "and include explicit user-aligned defaults before advancing phases."
        )
    return (
        "Policy: stable performance. Maintain strict validation and keep responses concise, "
        "reproducible, and policy-compliant."
    )


def reward_from_validation(valid: bool, warning_count: int, error_count: int) -> float:
    if not valid:
        return -1.0 - (0.1 * error_count)
    return max(0.1, 1.0 - (0.05 * warning_count))


def reward_from_user_decision(decision: str) -> float:
    if decision == "confirm":
        return 0.2
    return -0.1


def reward_from_runtime(success: bool, retry_count: int) -> float:
    if not success:
        return -0.5 - (0.1 * retry_count)
    return max(0.2, 1.0 - (0.1 * retry_count))


def reward_from_phase_latency(duration_sec: float) -> float:
    if duration_sec <= 1.0:
        return 0.3
    if duration_sec <= 5.0:
        return 0.15
    if duration_sec <= 30.0:
        return 0.0
    return -0.2


def reward_from_terminal_status(status: str, retry_count: int) -> float:
    normalized = str(status).lower()
    if normalized == "success":
        return max(0.5, 1.5 - (0.1 * retry_count))
    if normalized == "aborted":
        return -0.4 - (0.1 * retry_count)
    if normalized == "failed":
        return -0.8 - (0.1 * retry_count)
    return -0.2


def reward_from_evaluation(
    primary_metric: float,
    retry_count: int,
    runtime_sec: float,
    confirmations_requested: int,
) -> float:
    reward = 0.0

    # Metric quality.
    if primary_metric >= 0.9:
        reward += 1.0
    elif primary_metric >= 0.75:
        reward += 0.6
    elif primary_metric >= 0.6:
        reward += 0.2
    else:
        reward -= 0.6

    # Penalize retries to encourage stable first-pass execution.
    reward -= 0.15 * max(0, retry_count)

    # Penalize excessive latency.
    if runtime_sec > 300:
        reward -= 0.4
    elif runtime_sec > 120:
        reward -= 0.2
    elif runtime_sec > 30:
        reward -= 0.05

    # Penalize too many confirmations (high orchestration friction).
    if confirmations_requested > 2:
        reward -= 0.1 * (confirmations_requested - 2)

    return reward
