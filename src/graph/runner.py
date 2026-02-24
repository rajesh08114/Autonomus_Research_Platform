from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

from src.agents.clarifier_agent import (
    QUESTION_BANK,
    build_clarification_question,
    clarifier_agent_node,
    next_clarification_question_id,
)
from src.agents.code_gen_agent import code_gen_agent_node
from src.agents.dataset_agent import dataset_agent_node
from src.agents.doc_generator_agent import doc_generator_agent_node
from src.agents.env_manager_agent import apply_user_confirmation, env_manager_agent_node
from src.agents.error_recovery_agent import error_recovery_agent_node
from src.agents.evaluator_agent import evaluator_agent_node
from src.agents.job_scheduler_agent import job_scheduler_agent_node
from src.agents.planner_agent import planner_agent_node
from src.agents.quantum_gate import quantum_gate_node
from src.config.settings import settings
from src.core.logger import get_logger
from src.core.phase_validator import validate_phase_output
from src.core.rl_feedback import (
    record_phase_feedback,
    reward_from_evaluation,
    reward_from_phase_latency,
    reward_from_runtime,
    reward_from_terminal_status,
    reward_from_user_decision,
    reward_from_validation,
)
from src.core.subprocess_runner import subprocess_runner_node
from src.db.repository import ExperimentRepository
from src.state.research_state import ExperimentStatus, ResearchState, new_research_state

logger = get_logger(__name__)

_RUN_SEMAPHORE = asyncio.Semaphore(max(1, int(settings.MAX_CONCURRENT_EXPS)))
_RUN_TASKS: dict[str, asyncio.Task] = {}


def _new_experiment_id() -> str:
    date = time.strftime("%Y%m%d")
    return f"exp_{date}_{uuid.uuid4().hex[:6]}"


def _is_waiting(state: ResearchState) -> bool:
    status = state["status"]
    value = status.value if hasattr(status, "value") else str(status)
    return value == ExperimentStatus.WAITING.value


def _is_terminal(state: ResearchState) -> bool:
    status = state["status"]
    value = status.value if hasattr(status, "value") else str(status)
    return value in {ExperimentStatus.SUCCESS.value, ExperimentStatus.ABORTED.value, ExperimentStatus.FAILED.value}


def _apply_answers(state: ResearchState, answers: dict[str, Any]) -> None:
    mapped: dict[str, Any] = {}
    for qid, value in answers.items():
        item = QUESTION_BANK.get(qid)
        if item:
            mapped[item["topic"]] = value
        else:
            mapped[qid] = value
    state["clarifications"].update(mapped)


def _active_question(pending: dict[str, Any]) -> dict[str, Any] | None:
    current = pending.get("current_question")
    if isinstance(current, dict):
        return current
    questions = pending.get("questions", [])
    if isinstance(questions, list) and questions and isinstance(questions[0], dict):
        return questions[0]
    return None


def _should_fail_injection(point: str) -> bool:
    if not settings.FAILURE_INJECTION_ENABLED:
        return False
    configured = {p.strip().lower() for p in settings.FAILURE_INJECTION_POINTS.split(",") if p.strip()}
    if configured and point.lower() not in configured:
        return False
    return random.random() < float(max(0.0, min(1.0, settings.FAILURE_INJECTION_RATE)))


def _running_task(experiment_id: str) -> asyncio.Task | None:
    task = _RUN_TASKS.get(experiment_id)
    if task and task.done():
        _RUN_TASKS.pop(experiment_id, None)
        return None
    return task


def _schedule_background_run(experiment_id: str) -> None:
    if _running_task(experiment_id):
        return
    task = asyncio.create_task(_run_background(experiment_id))
    _RUN_TASKS[experiment_id] = task


async def _run_background(experiment_id: str) -> None:
    async with _RUN_SEMAPHORE:
        state = await ExperimentRepository.get(experiment_id)
        if not state:
            return
        try:
            await run_until_blocked(state)
        except Exception as exc:
            logger.exception("workflow.background.error", experiment_id=experiment_id)
            await ExperimentRepository.mark_failed(experiment_id, str(exc))


async def start_experiment(prompt: str, config_overrides: dict[str, Any]) -> str:
    experiment_id = _new_experiment_id()
    project_path = str((settings.project_root_path / experiment_id).resolve())
    Path(project_path).mkdir(parents=True, exist_ok=True)

    state = new_research_state(experiment_id, project_path, prompt, config_overrides)
    provider = settings.MASTER_LLM_PROVIDER.strip().lower()
    state["llm_provider"] = provider
    state["llm_model"] = settings.huggingface_model_id if provider in {"huggingface", "hf", "hugging_face"} else settings.MASTER_LLM_MODEL
    state["execution_target"] = "local_machine"
    logger.info("experiment.start", experiment_id=experiment_id, project_path=project_path)
    state = await clarifier_agent_node(state)
    validation = validate_phase_output("clarifier", state)
    await record_phase_feedback(
        experiment_id=experiment_id,
        phase="clarifier",
        reward=reward_from_validation(validation.ok, len(validation.warnings), len(validation.errors)),
        signal="phase_validation",
        details={"errors": validation.errors[:5], "warnings": validation.warnings[:5]},
    )
    if not validation.ok:
        logger.error("experiment.start.validation_failed", experiment_id=experiment_id, errors=validation.errors)
        state["status"] = ExperimentStatus.FAILED.value
        state["errors"].append(
            {
                "category": "ValidationError",
                "message": "; ".join(validation.errors[:3]),
                "file_path": "clarifier",
                "line_number": 0,
                "traceback": "; ".join(validation.errors),
                "timestamp": time.time(),
            }
        )
    await ExperimentRepository.create(state)
    await ExperimentRepository.add_log(
        experiment_id,
        "clarifier",
        "info",
        "Clarification questions generated",
        {
            "current_question": (state.get("pending_user_question") or {}).get("current_question"),
            "asked_question_ids": (state.get("pending_user_question") or {}).get("asked_question_ids", []),
            "mode": (state.get("pending_user_question") or {}).get("mode"),
        },
    )
    await ExperimentRepository.add_log(
        experiment_id,
        "system",
        "info",
        "Experiment configured for local execution",
        {"execution_target": state["execution_target"], "llm_provider": state["llm_provider"], "llm_model": state["llm_model"]},
    )
    return experiment_id


async def run_until_blocked(state: ResearchState) -> ResearchState:
    for _ in range(100):
        if _is_waiting(state) or _is_terminal(state):
            break

        phase = state["phase"]
        started = time.time()
        should_stop = False
        logger.info("phase.start", experiment_id=state["experiment_id"], phase=phase, retry_count=state.get("retry_count", 0))
        await ExperimentRepository.add_log(state["experiment_id"], phase, "info", f"Phase {phase} started")

        if phase != "error_recovery" and _should_fail_injection(phase):
            injected_message = f"Injected failure at phase {phase}"
            state["errors"].append(
                {
                    "category": "InjectedFailure",
                    "message": injected_message,
                    "file_path": phase,
                    "line_number": 0,
                    "traceback": injected_message,
                    "timestamp": time.time(),
                }
            )
            await ExperimentRepository.add_log(
                state["experiment_id"],
                phase,
                "error",
                "Injected phase failure",
                {"phase": phase, "category": "InjectedFailure"},
            )
            state["phase"] = "error_recovery"
            await ExperimentRepository.update(state["experiment_id"], state)
            continue

        if phase == "planner":
            state = await planner_agent_node(state)
            state["phase"] = "env_manager"
        elif phase == "env_manager":
            state = await env_manager_agent_node(state)
            if state.get("pending_user_confirm"):
                await ExperimentRepository.add_log(
                    state["experiment_id"],
                    "env_manager",
                    "info",
                    "User confirmation required",
                    {"pending_action": state.get("pending_user_confirm")},
                )
            if not _is_waiting(state):
                state["phase"] = "dataset_manager"
        elif phase == "dataset_manager":
            state = await dataset_agent_node(state)
            state["phase"] = "code_generator"
        elif phase == "code_generator":
            state = await code_gen_agent_node(state)
            state["phase"] = "quantum_gate" if state["requires_quantum"] else "job_scheduler"
        elif phase == "quantum_gate":
            state = await quantum_gate_node(state)
            state["phase"] = "job_scheduler"
        elif phase == "job_scheduler":
            state = await job_scheduler_agent_node(state)
            state["phase"] = "subprocess_runner"
        elif phase == "subprocess_runner":
            state = await subprocess_runner_node(state)
            if state["execution_logs"] and state["execution_logs"][-1]["returncode"] != 0:
                state["phase"] = "error_recovery"
            elif len(state["execution_logs"]) >= len(state["execution_order"]):
                state["phase"] = "results_evaluator"
            else:
                state["phase"] = "subprocess_runner"
        elif phase == "error_recovery":
            state = await error_recovery_agent_node(state)
            if _is_terminal(state):
                should_stop = True
            if state["phase"] not in {"env_manager", "subprocess_runner"}:
                state["phase"] = "subprocess_runner"
        elif phase == "results_evaluator":
            state = await evaluator_agent_node(state)
            primary_name = str(state.get("target_metric", "accuracy"))
            evaluation = (state.get("metrics") or {}).get("evaluation", {})
            primary_metric = float((evaluation or {}).get(primary_name, 0.0))
            if (
                settings.AUTO_RETRY_ON_LOW_METRIC
                and primary_metric < float(settings.MIN_PRIMARY_METRIC_FOR_SUCCESS)
                and int(state.get("retry_count", 0)) < int(settings.MAX_RETRY_COUNT)
            ):
                state["retry_count"] = int(state.get("retry_count", 0)) + 1
                state["max_epochs"] = int(state.get("max_epochs") or 20) + 10
                state["phase"] = "subprocess_runner"
                await ExperimentRepository.add_log(
                    state["experiment_id"],
                    "results_evaluator",
                    "warning",
                    "Auto-retry scheduled due to low primary metric",
                    {
                        "target_metric": primary_name,
                        "primary_metric": primary_metric,
                        "threshold": float(settings.MIN_PRIMARY_METRIC_FOR_SUCCESS),
                        "retry_count": state["retry_count"],
                        "next_phase": "subprocess_runner",
                    },
                )
                await record_phase_feedback(
                    experiment_id=state["experiment_id"],
                    phase="results_evaluator",
                    reward=-0.4,
                    signal="auto_retry_decision",
                    details={
                        "target_metric": primary_name,
                        "primary_metric": primary_metric,
                        "threshold": float(settings.MIN_PRIMARY_METRIC_FOR_SUCCESS),
                    },
                )
            else:
                state["phase"] = "doc_generator"
        elif phase == "doc_generator":
            state = await doc_generator_agent_node(state)
            should_stop = True
        else:
            break

        validation = validate_phase_output(phase, state)
        await record_phase_feedback(
            experiment_id=state["experiment_id"],
            phase=phase,
            reward=reward_from_validation(validation.ok, len(validation.warnings), len(validation.errors)),
            signal="phase_validation",
            details={"errors": validation.errors[:5], "warnings": validation.warnings[:5]},
        )
        for warning in validation.warnings:
            logger.warning("phase.validation.warning", experiment_id=state["experiment_id"], phase=phase, warning=warning)
            await ExperimentRepository.add_log(state["experiment_id"], phase, "warning", warning)
        if not validation.ok:
            logger.error("phase.validation.failed", experiment_id=state["experiment_id"], phase=phase, errors=validation.errors)
            state["status"] = ExperimentStatus.FAILED.value
            state["errors"].append(
                {
                    "category": "ValidationError",
                    "message": "; ".join(validation.errors[:3]),
                    "file_path": phase,
                    "line_number": 0,
                    "traceback": "; ".join(validation.errors),
                    "timestamp": time.time(),
                }
            )
            await ExperimentRepository.add_log(
                state["experiment_id"],
                phase,
                "error",
                "Phase validation failed",
                {"errors": validation.errors},
            )
            await ExperimentRepository.update(state["experiment_id"], state)
            break

        if phase == "subprocess_runner":
            success = bool(state["execution_logs"] and state["execution_logs"][-1]["returncode"] == 0)
            await record_phase_feedback(
                experiment_id=state["experiment_id"],
                phase=phase,
                reward=reward_from_runtime(success=success, retry_count=state.get("retry_count", 0)),
                signal="runtime_outcome",
                details={"success": success, "retry_count": state.get("retry_count", 0)},
            )
        if phase == "results_evaluator":
            primary_name = str(state.get("target_metric", "accuracy"))
            evaluation = (state.get("metrics") or {}).get("evaluation", {})
            primary_metric = float((evaluation or {}).get(primary_name, 0.0))
            runtime_sec = float(state.get("phase_timings", {}).get("subprocess_runner", 0.0))
            confirmations_requested = int(state.get("confirmations_requested", 0))
            await record_phase_feedback(
                experiment_id=state["experiment_id"],
                phase=phase,
                reward=reward_from_evaluation(
                    primary_metric=primary_metric,
                    retry_count=int(state.get("retry_count", 0)),
                    runtime_sec=runtime_sec,
                    confirmations_requested=confirmations_requested,
                ),
                signal="evaluation_outcome",
                details={
                    "target_metric": primary_name,
                    "primary_metric": primary_metric,
                    "runtime_sec": runtime_sec,
                    "retry_count": int(state.get("retry_count", 0)),
                    "confirmations_requested": confirmations_requested,
                },
            )

        phase_duration = time.time() - started
        state["phase_timings"][phase] = state["phase_timings"].get(phase, 0.0) + phase_duration
        state["total_duration_sec"] = max(0.0, time.time() - float(state.get("timestamp_start") or time.time()))
        await record_phase_feedback(
            experiment_id=state["experiment_id"],
            phase=phase,
            reward=reward_from_phase_latency(phase_duration),
            signal="phase_latency",
            details={"duration_sec": round(phase_duration, 6)},
        )
        logger.info(
            "phase.end",
            experiment_id=state["experiment_id"],
            phase=phase,
            status=state["status"],
            duration_sec=round(phase_duration, 4),
        )
        await ExperimentRepository.update(state["experiment_id"], state)
        await ExperimentRepository.add_log(state["experiment_id"], phase, "info", f"Phase {phase} completed")
        if should_stop:
            break

    status = str(state.get("status", "unknown"))
    if _is_terminal(state):
        await record_phase_feedback(
            experiment_id=state["experiment_id"],
            phase="system",
            reward=reward_from_terminal_status(status, state.get("retry_count", 0)),
            signal="terminal_outcome",
            details={"status": status, "retry_count": state.get("retry_count", 0)},
        )
        if status.lower() == "success" and int(state.get("retry_count", 0)) > 0:
            await record_phase_feedback(
                experiment_id=state["experiment_id"],
                phase="error_recovery",
                reward=0.5,
                signal="recovery_success",
                details={"retry_count": int(state.get("retry_count", 0))},
            )

    llm_totals = await ExperimentRepository.get_experiment_llm_totals(state["experiment_id"])
    state["total_tokens_used"] = int(llm_totals.get("total_tokens", 0.0))
    state["llm_total_cost_usd"] = float(llm_totals.get("total_cost", 0.0))
    state["total_duration_sec"] = max(0.0, time.time() - float(state.get("timestamp_start") or time.time()))

    await ExperimentRepository.update(state["experiment_id"], state)
    return state


async def submit_answers(experiment_id: str, answers: dict[str, Any]) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    status = state["status"]
    status_value = status.value if hasattr(status, "value") else str(status)
    if status_value != ExperimentStatus.WAITING.value or not state.get("pending_user_question"):
        raise RuntimeError("Experiment is not waiting for clarification answers")

    pending = state.get("pending_user_question") or {}
    current = _active_question(pending)
    if not current:
        raise RuntimeError("No active clarification question found")
    if len(answers) != 1:
        raise RuntimeError("Submit exactly one answer per request")

    qid, value = next(iter(answers.items()))
    expected_qid = str(current.get("id", ""))
    if qid != expected_qid:
        raise RuntimeError(f"Expected answer for question {expected_qid}, got {qid}")

    _apply_answers(state, {qid: value})
    logger.info("experiment.answer_received", experiment_id=experiment_id, question_id=qid)

    answered = list(pending.get("answered") or [])
    answered.append({"id": qid, "value": value, "timestamp": time.time()})
    asked_question_ids = [str(x) for x in list(pending.get("asked_question_ids") or []) if x]
    if qid not in asked_question_ids:
        asked_question_ids.append(qid)

    next_qid = next_clarification_question_id(state, asked_question_ids=asked_question_ids)
    max_questions = int(pending.get("max_questions") or 12)
    if len(asked_question_ids) >= max_questions:
        next_qid = None

    if next_qid:
        next_question = build_clarification_question(
            next_qid,
            state,
            prompt_flags=pending.get("prompt_flags") if isinstance(pending.get("prompt_flags"), dict) else None,
        )
        state["pending_user_question"] = {
            "mode": str(pending.get("mode") or "sequential_dynamic"),
            "current_question": next_question,
            "questions": [next_question],
            "asked_question_ids": asked_question_ids,
            "answered": answered,
            "answered_count": len(answered),
            "total_questions_planned": len(answered) + 1,
            "max_questions": max_questions,
            "prompt_flags": pending.get("prompt_flags", {}),
        }
        state["status"] = ExperimentStatus.WAITING.value
        state["phase"] = "clarifier"
        await ExperimentRepository.update(experiment_id, state)
        await ExperimentRepository.add_log(
            experiment_id,
            "clarifier",
            "info",
            "Clarification answer received; next question queued",
            {
                "answer": {"id": qid, "value": value},
                "next_question": next_question,
                "asked_question_ids": asked_question_ids,
            },
        )
        return state

    state["pending_user_question"] = None
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "planner"
    await ExperimentRepository.update(experiment_id, state)
    await ExperimentRepository.add_log(
        experiment_id,
        "clarifier",
        "info",
        "Clarification completed",
        {"answers": answered},
    )
    if settings.WORKFLOW_BACKGROUND_ENABLED:
        _schedule_background_run(experiment_id)
        return state
    return await run_until_blocked(state)


async def submit_confirmation(
    experiment_id: str,
    action_id: str,
    decision: str,
    reason: str = "",
    alternative_preference: str = "",
) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    if not state.get("pending_user_confirm"):
        raise RuntimeError("No pending confirmation")

    state = await apply_user_confirmation(state, action_id, decision, reason, alternative_preference)
    logger.info("experiment.confirmation", experiment_id=experiment_id, action_id=action_id, decision=decision)
    await record_phase_feedback(
        experiment_id=experiment_id,
        phase="env_manager",
        reward=reward_from_user_decision(decision),
        signal="user_confirmation",
        details={"decision": decision, "action_id": action_id},
    )
    await ExperimentRepository.update(experiment_id, state)
    await ExperimentRepository.add_log(
        experiment_id,
        "env_manager",
        "info",
        f"Confirmation processed: {decision}",
        {"action_id": action_id, "decision": decision, "reason": reason, "alternative_preference": alternative_preference},
    )
    if not _is_waiting(state):
        state["phase"] = "env_manager"
        await ExperimentRepository.update(experiment_id, state)
        if settings.WORKFLOW_BACKGROUND_ENABLED:
            _schedule_background_run(experiment_id)
            return state
        state = await run_until_blocked(state)
    return state


async def abort_experiment(experiment_id: str, reason: str, save_partial: bool) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    if _is_terminal(state):
        return state
    logger.warning("experiment.abort", experiment_id=experiment_id, reason=reason, save_partial=save_partial)

    state["status"] = ExperimentStatus.ABORTED.value
    state["phase"] = "aborted"
    state["timestamp_end"] = time.time()

    project = Path(state["project_path"])
    docs = project / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    error_report = docs / "error_report.md"
    error_report.write_text(f"# Aborted\n\nReason: {reason}\n", encoding="utf-8")

    if save_partial:
        partial = project / "outputs" / "partial"
        partial.mkdir(parents=True, exist_ok=True)

    await ExperimentRepository.update(experiment_id, state)
    await ExperimentRepository.add_log(experiment_id, "system", "warning", "Experiment aborted", {"reason": reason})
    return state


async def retry_experiment(
    experiment_id: str,
    from_phase: str | None,
    reset_retries: bool,
    override_config: dict[str, Any] | None,
) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    logger.info("experiment.retry", experiment_id=experiment_id, from_phase=from_phase, reset_retries=reset_retries)

    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = from_phase or "error_recovery"
    if reset_retries:
        state["retry_count"] = 0
    if override_config:
        if "hardware_target" in override_config:
            state["hardware_target"] = override_config["hardware_target"]
        if "max_epochs" in override_config:
            state["max_epochs"] = int(override_config["max_epochs"])

    await ExperimentRepository.update(experiment_id, state)
    await ExperimentRepository.add_log(experiment_id, "system", "info", "Experiment retry initiated")
    if settings.WORKFLOW_BACKGROUND_ENABLED:
        _schedule_background_run(experiment_id)
        return state
    return await run_until_blocked(state)


async def get_experiment_or_404(experiment_id: str) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    return state


def state_progress_pct(state: ResearchState) -> int:
    mapping = {
        "clarifier": 10,
        "planner": 20,
        "env_manager": 30,
        "dataset_manager": 45,
        "code_generator": 60,
        "quantum_gate": 70,
        "job_scheduler": 75,
        "subprocess_runner": 85,
        "results_evaluator": 92,
        "doc_generator": 97,
        "finished": 100,
        "aborted": 100,
    }
    return mapping.get(state["phase"], 0)


def summarize_for_list(state_row: dict[str, Any]) -> dict[str, Any]:
    duration = None
    if state_row.get("completed_at"):
        duration = 0
    return {
        "experiment_id": state_row["id"],
        "status": state_row["status"],
        "phase": state_row["phase"],
        "prompt_preview": (state_row["prompt"] or "")[:80],
        "requires_quantum": bool(state_row["requires_quantum"]),
        "framework": state_row.get("framework"),
        "primary_metric": {"name": state_row.get("target_metric"), "value": None},
        "created_at": state_row.get("created_at"),
        "duration_sec": duration,
    }
