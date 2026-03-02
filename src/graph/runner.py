from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

from src.agents.clarifier_agent import clarifier_agent_node, coerce_answer_value, regenerate_pending_question_state
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
from src.core.execution_mode import is_vscode_execution_mode, normalize_execution_mode
from src.core.history_scope import build_collection_key, normalize_user_id
from src.core.logger import get_logger
from src.core.phase_validator import validate_phase_output
from src.core.prompt_domain import validate_supported_prompt
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
from src.core.webhooks import emit_webhook_event
from src.db.repository import ExperimentRepository
from src.state.research_state import ExperimentStatus, ResearchState, new_research_state

logger = get_logger(__name__)

_RUN_SEMAPHORE = asyncio.Semaphore(max(1, int(settings.MAX_CONCURRENT_EXPS)))
_RUN_TASKS: dict[str, asyncio.Task] = {}
_EXPERIMENT_LOCKS: dict[str, asyncio.Lock] = {}
_PATCHABLE_FIELDS = {
    "user_prompt",
    "research_type",
    "framework",
    "dataset_source",
    "target_metric",
    "hardware_target",
    "output_format",
    "max_epochs",
    "batch_size",
    "random_seed",
    "kaggle_dataset_id",
    "clarifications",
    "research_plan",
    "requires_quantum",
    "quantum_framework",
    "quantum_algorithm",
    "quantum_backend",
    "quantum_qubit_count",
    "default_allow_research",
}
_PATCHABLE_STR_FIELDS = {
    "user_prompt",
    "research_type",
    "framework",
    "dataset_source",
    "target_metric",
    "hardware_target",
    "output_format",
    "kaggle_dataset_id",
    "quantum_framework",
    "quantum_algorithm",
    "quantum_backend",
}
_PATCHABLE_INT_FIELDS = {"max_epochs", "batch_size", "random_seed", "quantum_qubit_count"}


def _experiment_lock(experiment_id: str) -> asyncio.Lock:
    lock = _EXPERIMENT_LOCKS.get(experiment_id)
    if lock is None:
        lock = asyncio.Lock()
        _EXPERIMENT_LOCKS[experiment_id] = lock
    return lock


async def _with_experiment_lock(experiment_id: str, fn: Any) -> Any:
    lock = _experiment_lock(experiment_id)
    async with lock:
        return await fn()


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


def _is_terminal_status(status: str) -> bool:
    return str(status or "") in {ExperimentStatus.SUCCESS.value, ExperimentStatus.ABORTED.value, ExperimentStatus.FAILED.value}


def _state_value(state: ResearchState, key: str) -> str:
    value = state.get(key, "")
    return value.value if hasattr(value, "value") else str(value)


async def _emit_state_webhook(state: ResearchState, event: str, data: dict[str, Any] | None = None) -> None:
    await emit_webhook_event(
        webhook_url=str(state.get("webhook_url") or ""),
        event=event,
        experiment_id=str(state.get("experiment_id") or ""),
        status=_state_value(state, "status"),
        phase=_state_value(state, "phase"),
        data=data or {},
    )


async def _emit_transition_webhooks(
    state: ResearchState,
    previous_status: str,
    previous_phase: str,
) -> None:
    current_status = _state_value(state, "status")
    current_phase = _state_value(state, "phase")

    if previous_status != current_status:
        await _emit_state_webhook(
            state,
            "experiment.status_changed",
            {"previous_status": previous_status, "status": current_status},
        )
    if previous_phase != current_phase:
        await _emit_state_webhook(
            state,
            "experiment.phase_changed",
            {"previous_phase": previous_phase, "phase": current_phase},
        )

    if current_status == ExperimentStatus.WAITING.value:
        pending_confirm = state.get("pending_user_confirm")
        if pending_confirm:
            await _emit_state_webhook(
                state,
                "confirmation.required",
                {"pending_action": pending_confirm},
            )
        pending_question = state.get("pending_user_question")
        if pending_question:
            await _emit_state_webhook(
                state,
                "clarifier.question_required",
                {"pending_question": pending_question},
            )

    if not _is_terminal_status(previous_status) and _is_terminal_status(current_status):
        await _emit_state_webhook(
            state,
            "experiment.completed",
            {
                "outcome": current_status,
                "metrics": state.get("metrics", {}),
                "documentation_path": state.get("documentation_path"),
            },
        )
        if current_status == ExperimentStatus.SUCCESS.value and state.get("documentation_path"):
            await _emit_state_webhook(
                state,
                "report.ready",
                {"report_path": state.get("documentation_path")},
            )


async def _emit_phase_status_webhook(
    state: ResearchState,
    phase: str,
    outcome: str,
    duration_sec: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "phase": str(phase or ""),
        "outcome": str(outcome or ""),
        "status": _state_value(state, "status"),
    }
    if duration_sec is not None:
        payload["duration_sec"] = round(float(duration_sec), 6)
    if details:
        payload.update(details)
    await _emit_state_webhook(state, "phase.status", payload)


def _question_plan(pending: dict[str, Any]) -> list[dict[str, Any]]:
    plan = pending.get("question_plan")
    if isinstance(plan, list):
        output = [item for item in plan if isinstance(item, dict)]
        if output:
            return output
    current = _active_question(pending)
    return [current] if isinstance(current, dict) else []


def _question_topic(question: dict[str, Any]) -> str:
    topic = str(question.get("topic", "")).strip().lower()
    if topic:
        return topic
    return str(question.get("id", "unknown")).strip() or "unknown"


def _apply_answers(state: ResearchState, answers: dict[str, Any], current_question: dict[str, Any]) -> None:
    topic = _question_topic(current_question)
    _, value = next(iter(answers.items()))
    state["clarifications"][topic] = coerce_answer_value(current_question, value)


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


def _is_recoverable_codegen_error(phase: str, exc: Exception) -> bool:
    if str(phase or "").strip().lower() != "code_generator":
        return False
    message = str(exc or "").strip().lower()
    if not message:
        return False
    signals = (
        "strict_state_only violation",
        "dynamic code generation failed",
        "payload problem_type mismatch",
        "missing required files",
        "payload code_level mismatch",
        "payload algorithm_class mismatch",
    )
    return any(signal in message for signal in signals)


def _bool_from_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in {0, 1}:
            return bool(value)
        return None
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled", "disable"}:
        return False
    return None


def _resolve_default_allow_research(overrides: dict[str, Any] | None) -> bool:
    payload = dict(overrides or {})
    for key in (
        "default_allow_research",
        "default_allow_auto_research",
        "allow_auto_research",
        "auto_research",
    ):
        parsed = _bool_from_value(payload.get(key))
        if parsed is not None:
            return parsed
    return bool(settings.DEFAULT_ALLOW_RESEARCH)


def _default_allow_research_enabled(state: ResearchState) -> bool:
    parsed = _bool_from_value(state.get("default_allow_research"))
    if parsed is not None:
        return parsed
    return bool(settings.DEFAULT_ALLOW_RESEARCH)


def _default_answer_for_question(state: ResearchState, question: dict[str, Any]) -> Any:
    topic = _question_topic(question)
    qtype = str(question.get("type", "text")).strip().lower()
    raw_default = question.get("default")

    if raw_default is not None:
        if not isinstance(raw_default, str) or raw_default.strip():
            return raw_default

    options = [str(opt).strip() for opt in question.get("options", []) if str(opt).strip()]

    if qtype == "choice":
        if topic == "output_format":
            preferred = str(state.get("output_format") or "").strip().lower()
            if preferred in {".py", ".ipynb", "hybrid"} and preferred in options:
                return preferred
        if options:
            return options[0]
        if topic == "output_format":
            preferred = str(state.get("output_format") or "").strip().lower()
            if preferred in {".py", ".ipynb", "hybrid"}:
                return preferred
            return "hybrid"
        if topic == "framework_preference":
            return str(state.get("framework") or "auto")
        return "auto"

    if qtype == "boolean":
        if topic == "requires_quantum":
            return bool(str(state.get("research_type", "ai")).strip().lower() == "quantum")
        return True

    if qtype == "number":
        if topic in {"max_epochs", "epochs"}:
            return int(state.get("max_epochs") or 50)
        if topic == "batch_size":
            return int(state.get("batch_size") or 32)
        if topic in {"random_seed", "seed"}:
            return int(state.get("random_seed") or 42)
        return 1

    if "path" in topic:
        if "dataset" in topic and str(state.get("dataset_path") or "").strip():
            return str(state.get("dataset_path"))
        return str(state.get("project_path") or "")
    if topic == "target_metric":
        return str(state.get("target_metric") or "accuracy")
    if topic == "research_type":
        return str(state.get("research_type") or "ai")
    return "auto"


async def _auto_process_pending_questions(state: ResearchState) -> bool:
    if not _default_allow_research_enabled(state):
        return False

    pending = state.get("pending_user_question") or {}
    if not pending:
        return False

    current = _active_question(pending)
    if not current:
        state["pending_user_question"] = None
        if not _is_terminal(state):
            state["status"] = ExperimentStatus.RUNNING.value
            if str(state.get("phase", "")).strip() == "clarifier":
                state["phase"] = "planner"
        return True

    try:
        qid = str(current.get("id", "")).strip() or "Q_AUTO"
        answer = _default_answer_for_question(state, current)
        _apply_answers(state, {qid: answer}, current)
        normalized_value = (state.get("clarifications") or {}).get(_question_topic(current))

        answered = list(pending.get("answered") or [])
        answered.append({"id": qid, "topic": _question_topic(current), "value": normalized_value, "timestamp": time.time()})
        asked_question_ids = [str(x) for x in list(pending.get("asked_question_ids") or []) if x]
        if qid not in asked_question_ids:
            asked_question_ids.append(qid)

        pending_for_replan = dict(pending)
        pending_for_replan["asked_question_ids"] = asked_question_ids
        pending_for_replan["answered"] = answered
        next_pending = await regenerate_pending_question_state(state, pending_for_replan)
        state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1

        if next_pending:
            state["pending_user_question"] = next_pending
            state["status"] = ExperimentStatus.WAITING.value
            state["phase"] = "clarifier"
            await ExperimentRepository.add_log(
                state["experiment_id"],
                "clarifier",
                "info",
                "Auto-answered clarification question",
                {"question_id": qid, "answer": normalized_value},
            )
            return True

        state["pending_user_question"] = None
        state["status"] = ExperimentStatus.RUNNING.value
        if str(state.get("phase", "")).strip() in {"clarifier", ""}:
            state["phase"] = "planner"
        await ExperimentRepository.add_log(
            state["experiment_id"],
            "clarifier",
            "info",
            "Clarification auto-completed",
            {"answered_count": len(answered), "default_allow_research": True},
        )
        return True
    except Exception as exc:
        logger.warning(
            "clarifier.auto_answer_failed",
            experiment_id=state.get("experiment_id"),
            error=str(exc),
        )
        await ExperimentRepository.add_log(
            state["experiment_id"],
            "clarifier",
            "warning",
            "Auto-answer failed; waiting for manual clarification",
            {"error": str(exc)[:500]},
        )
        return False


async def _auto_process_pending_confirmation(state: ResearchState) -> bool:
    if not _default_allow_research_enabled(state):
        return False
    if is_vscode_execution_mode(state):
        # VS Code mode requires extension-side local execution payloads.
        return False

    pending = state.get("pending_user_confirm") or {}
    if not pending:
        return False

    action_id = str(pending.get("action_id", "")).strip()
    if not action_id:
        state["pending_user_confirm"] = None
        if not _is_terminal(state):
            state["status"] = ExperimentStatus.RUNNING.value
        return True

    pending_phase = str(pending.get("phase", state.get("phase", "env_manager")))
    try:
        updated_state = await apply_user_confirmation(
            state,
            action_id=action_id,
            decision="confirm",
            reason="Auto-confirm enabled by default_allow_research",
            alternative_preference="",
            execution_result=None,
        )
        if updated_state is not state:
            state.clear()
            state.update(updated_state)
        await record_phase_feedback(
            experiment_id=state["experiment_id"],
            phase=pending_phase,
            reward=reward_from_user_decision("confirm"),
            signal="auto_user_confirmation",
            details={"action_id": action_id, "action": pending.get("action")},
        )
        await ExperimentRepository.add_log(
            state["experiment_id"],
            pending_phase,
            "info",
            "Auto-confirmation applied",
            {"action_id": action_id, "action": pending.get("action")},
        )
        return True
    except Exception as exc:
        logger.warning(
            "confirmation.auto_confirm_failed",
            experiment_id=state.get("experiment_id"),
            action_id=action_id,
            error=str(exc),
        )
        await ExperimentRepository.add_log(
            state["experiment_id"],
            pending_phase,
            "warning",
            "Auto-confirm failed; waiting for user confirmation",
            {"action_id": action_id, "error": str(exc)[:500]},
        )
        return False


def _normalize_patch_choice(value: str, allowed: set[str]) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return None


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
        lock = _experiment_lock(experiment_id)
        async with lock:
            state = await ExperimentRepository.get(experiment_id)
            if not state:
                return
            try:
                await run_until_blocked(state)
            except Exception as exc:
                logger.exception("workflow.background.error", experiment_id=experiment_id)
                await ExperimentRepository.mark_failed(experiment_id, str(exc))
                failed_state = await ExperimentRepository.get(experiment_id)
                if failed_state:
                    await _emit_state_webhook(
                        failed_state,
                        "experiment.completed",
                        {"outcome": ExperimentStatus.FAILED.value, "reason": str(exc)},
                    )


async def start_experiment(
    prompt: str,
    config_overrides: dict[str, Any],
    research_type: str = "ai",
    user_id: str | None = None,
    test_mode: bool = False,
    webhook_url: str | None = None,
) -> str:
    experiment_id = _new_experiment_id()
    overrides = dict(config_overrides or {})
    overrides["default_allow_research"] = _resolve_default_allow_research(overrides)
    execution_mode = normalize_execution_mode(overrides.get("execution_mode", settings.EXECUTION_MODE))
    project_root_override = str(overrides.get("project_root", "")).strip()
    if execution_mode == "vscode_extension":
        if project_root_override:
            project_root = Path(project_root_override).expanduser().resolve()
        else:
            configured_root = Path(settings.PROJECT_ROOT).expanduser()
            if configured_root.is_absolute():
                project_root = configured_root.resolve()
            else:
                project_root = (Path.home() / "ARP" / "workspace" / "projects").resolve()
        project_path = str((project_root / experiment_id).resolve())
    else:
        project_path = str((settings.project_root_path / experiment_id).resolve())
        Path(project_path).mkdir(parents=True, exist_ok=True)

    normalized_user = normalize_user_id(user_id)
    collection_key = build_collection_key(normalized_user, test_mode)
    overrides["user_id"] = normalized_user
    overrides["test_mode"] = bool(test_mode)
    overrides["collection_key"] = collection_key
    overrides["webhook_url"] = str(webhook_url or "").strip()
    normalized_research_type = str(research_type or overrides.get("research_type", "ai")).strip().lower()
    if normalized_research_type not in {"ai", "quantum"}:
        normalized_research_type = "ai"
    overrides["research_type"] = normalized_research_type

    state = new_research_state(experiment_id, project_path, prompt, overrides)
    state["research_type"] = normalized_research_type
    state["llm_provider"] = "huggingface"
    state["llm_model"] = settings.huggingface_model_id
    state["execution_target"] = "local_machine"
    state["execution_mode"] = execution_mode
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
        {
            "execution_target": state["execution_target"],
            "execution_mode": state["execution_mode"],
            "llm_provider": state["llm_provider"],
            "llm_model": state["llm_model"],
            "research_type": state.get("research_type", "ai"),
            "default_allow_research": bool(state.get("default_allow_research", False)),
        },
    )
    await ExperimentRepository.add_to_collection(
        experiment_id=experiment_id,
        collection_key=collection_key,
        user_id=normalized_user,
        test_mode=test_mode,
        metadata={"prompt": prompt[:200], "research_type": state.get("research_type", "ai")},
    )
    await _emit_state_webhook(
        state,
        "experiment.started",
        {
            "research_type": state.get("research_type", "ai"),
            "test_mode": bool(state.get("test_mode", False)),
            "user_id": state.get("research_user_id", "anonymous"),
            "default_allow_research": bool(state.get("default_allow_research", False)),
        },
    )
    await _emit_transition_webhooks(
        state,
        previous_status=ExperimentStatus.PENDING.value,
        previous_phase="clarifier",
    )
    if _default_allow_research_enabled(state):
        await ExperimentRepository.add_log(
            experiment_id,
            "system",
            "info",
            "Default allow research enabled; running automatic workflow",
            {"default_allow_research": True},
        )
        if settings.WORKFLOW_BACKGROUND_ENABLED:
            _schedule_background_run(experiment_id)
        else:
            await run_until_blocked(state)
    return experiment_id


async def run_until_blocked(state: ResearchState) -> ResearchState:
    for _ in range(100):
        if state.get("pending_user_question") and not _is_terminal(state):
            if await _auto_process_pending_questions(state):
                continue
            state["status"] = ExperimentStatus.WAITING.value
        if state.get("pending_user_confirm") and not _is_terminal(state):
            if await _auto_process_pending_confirmation(state):
                continue
            state["status"] = ExperimentStatus.WAITING.value
        if _is_waiting(state) or _is_terminal(state):
            break

        previous_status = _state_value(state, "status")
        previous_phase = _state_value(state, "phase")
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

        try:
            if phase == "planner":
                state = await planner_agent_node(state)
                if not _is_waiting(state) and not _is_terminal(state):
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
                if not _is_waiting(state):
                    state["phase"] = "code_generator"
            elif phase == "code_generator":
                state = await code_gen_agent_node(state)
                if not _is_waiting(state):
                    state["phase"] = "quantum_gate" if state["requires_quantum"] else "job_scheduler"
            elif phase == "quantum_gate":
                state = await quantum_gate_node(state)
                if not _is_waiting(state):
                    state["phase"] = "job_scheduler"
            elif phase == "job_scheduler":
                state = await job_scheduler_agent_node(state)
                if not _is_waiting(state) and not _is_terminal(state) and str(state.get("phase", "")).strip() == "job_scheduler":
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
                # Keep the recovery-selected phase when provided; only default
                # back to subprocess_runner if recovery left phase unchanged.
                if str(state.get("phase", "")).strip() == "error_recovery":
                    state["phase"] = "subprocess_runner"
            elif phase == "results_evaluator":
                state = await evaluator_agent_node(state)
                primary_name = str(state.get("target_metric", "accuracy"))
                evaluation = (state.get("metrics") or {}).get("evaluation", {})
                primary_metric = float((evaluation or {}).get(primary_name, 0.0))
                retry_pref = _bool_from_value((state.get("clarifications") or {}).get("auto_retry_preference"))
                auto_retry_enabled = retry_pref if retry_pref is not None else bool(settings.AUTO_RETRY_ON_LOW_METRIC)
                if (
                    auto_retry_enabled
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
                            "auto_retry_enabled": auto_retry_enabled,
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
                            "auto_retry_enabled": auto_retry_enabled,
                        },
                    )
                else:
                    state["phase"] = "doc_generator"
            elif phase == "doc_generator":
                state = await doc_generator_agent_node(state)
                should_stop = True
            else:
                break
        except Exception as exc:
            logger.exception("phase.execution.failed", experiment_id=state["experiment_id"], phase=phase)
            if _is_recoverable_codegen_error(phase, exc) and int(state.get("retry_count", 0)) < int(settings.MAX_RETRY_COUNT):
                state["errors"].append(
                    {
                        "category": "CodeGenerationStrictError",
                        "message": str(exc)[:1000],
                        "file_path": phase,
                        "line_number": 0,
                        "traceback": str(exc),
                        "timestamp": time.time(),
                    }
                )
                state["status"] = ExperimentStatus.RUNNING.value
                state["phase"] = "error_recovery"
                await ExperimentRepository.add_log(
                    state["experiment_id"],
                    phase,
                    "warning",
                    "Code generation failed strict contract; redirecting to error recovery",
                    {"error": str(exc)},
                )
                phase_duration = time.time() - started
                await _emit_phase_status_webhook(
                    state,
                    phase=phase,
                    outcome="error",
                    duration_sec=phase_duration,
                    details={"error": str(exc)[:500], "recovery_redirect": True},
                )
                await ExperimentRepository.update(state["experiment_id"], state)
                await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
                continue
            state["status"] = ExperimentStatus.FAILED.value
            state["errors"].append(
                {
                    "category": "PhaseExecutionError",
                    "message": str(exc)[:1000],
                    "file_path": phase,
                    "line_number": 0,
                    "traceback": str(exc),
                    "timestamp": time.time(),
                }
            )
            await ExperimentRepository.add_log(
                state["experiment_id"],
                phase,
                "error",
                "Phase execution failed",
                {"error": str(exc)},
            )
            phase_duration = time.time() - started
            await _emit_phase_status_webhook(
                state,
                phase=phase,
                outcome="error",
                duration_sec=phase_duration,
                details={"error": str(exc)[:500]},
            )
            await ExperimentRepository.update(state["experiment_id"], state)
            await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
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
            phase_duration = time.time() - started
            await _emit_phase_status_webhook(
                state,
                phase=phase,
                outcome="error",
                duration_sec=phase_duration,
                details={"errors": validation.errors[:5]},
            )
            await ExperimentRepository.update(state["experiment_id"], state)
            await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
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
        phase_outcome = "waiting_user" if _is_waiting(state) else "completed"
        await _emit_phase_status_webhook(
            state,
            phase=phase,
            outcome=phase_outcome,
            duration_sec=phase_duration,
            details={
                "next_phase": str(state.get("phase", "")),
                "pending_action": bool(state.get("pending_user_confirm")),
                "pending_question": bool(state.get("pending_user_question")),
            },
        )
        await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
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
    async def _op() -> ResearchState:
        return await _submit_answers_unlocked(experiment_id, answers)

    return await _with_experiment_lock(experiment_id, _op)


async def _submit_answers_unlocked(experiment_id: str, answers: dict[str, Any]) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    previous_status = _state_value(state, "status")
    previous_phase = _state_value(state, "phase")
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

    _apply_answers(state, {qid: value}, current)
    normalized_value = (state.get("clarifications") or {}).get(_question_topic(current))
    logger.info("experiment.answer_received", experiment_id=experiment_id, question_id=qid)
    await _emit_state_webhook(
        state,
        "clarifier.answer_received",
        {"question_id": qid, "value": normalized_value},
    )

    answered = list(pending.get("answered") or [])
    answered.append({"id": qid, "topic": _question_topic(current), "value": normalized_value, "timestamp": time.time()})
    asked_question_ids = [str(x) for x in list(pending.get("asked_question_ids") or []) if x]
    if qid not in asked_question_ids:
        asked_question_ids.append(qid)

    pending_for_replan = dict(pending)
    pending_for_replan["asked_question_ids"] = asked_question_ids
    pending_for_replan["answered"] = answered
    next_pending = await regenerate_pending_question_state(state, pending_for_replan)
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1

    if next_pending:
        state["pending_user_question"] = next_pending
        state["status"] = ExperimentStatus.WAITING.value
        state["phase"] = "clarifier"
        await ExperimentRepository.update(experiment_id, state)
        await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
        await ExperimentRepository.add_log(
            experiment_id,
            "clarifier",
            "info",
            "Clarification answer received; next question queued",
            {
                "answer": {"id": qid, "value": normalized_value},
                "next_question": next_pending.get("current_question"),
                "asked_question_ids": next_pending.get("asked_question_ids", []),
            },
        )
        return state

    state["pending_user_question"] = None
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "planner"
    await ExperimentRepository.update(experiment_id, state)
    await _emit_state_webhook(
        state,
        "clarifier.completed",
        {"answered_count": len(answered), "answered": answered},
    )
    await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
    await ExperimentRepository.add_log(
        experiment_id,
        "clarifier",
        "info",
        "Clarification completed",
        {"answers": answered, "last_answer": {"id": qid, "value": normalized_value}},
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
    execution_result: dict[str, Any] | None = None,
) -> ResearchState:
    async def _op() -> ResearchState:
        return await _submit_confirmation_unlocked(
            experiment_id=experiment_id,
            action_id=action_id,
            decision=decision,
            reason=reason,
            alternative_preference=alternative_preference,
            execution_result=execution_result,
        )

    return await _with_experiment_lock(experiment_id, _op)


async def _submit_confirmation_unlocked(
    experiment_id: str,
    action_id: str,
    decision: str,
    reason: str = "",
    alternative_preference: str = "",
    execution_result: dict[str, Any] | None = None,
) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    previous_status = _state_value(state, "status")
    previous_phase = _state_value(state, "phase")
    if not state.get("pending_user_confirm"):
        raise RuntimeError("No pending confirmation")
    pending = state.get("pending_user_confirm") or {}
    pending_phase = str(pending.get("phase", state.get("phase", "env_manager")))

    state = await apply_user_confirmation(
        state,
        action_id,
        decision,
        reason,
        alternative_preference,
        execution_result=execution_result,
    )
    logger.info("experiment.confirmation", experiment_id=experiment_id, action_id=action_id, decision=decision)
    await record_phase_feedback(
        experiment_id=experiment_id,
        phase=pending_phase,
        reward=reward_from_user_decision(decision),
        signal="user_confirmation",
        details={"decision": decision, "action_id": action_id, "action": pending.get("action")},
    )
    await ExperimentRepository.update(experiment_id, state)
    await _emit_state_webhook(
        state,
        "confirmation.processed",
        {
            "action_id": action_id,
            "decision": decision,
            "reason": reason,
            "alternative_preference": alternative_preference,
        },
    )
    await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
    await ExperimentRepository.add_log(
        experiment_id,
        pending_phase,
        "info",
        f"Confirmation processed: {decision}",
        {
            "action_id": action_id,
            "decision": decision,
            "reason": reason,
            "alternative_preference": alternative_preference,
            "action": pending.get("action"),
        },
    )
    if not _is_waiting(state):
        await ExperimentRepository.update(experiment_id, state)
        if settings.WORKFLOW_BACKGROUND_ENABLED:
            _schedule_background_run(experiment_id)
            return state
        state = await run_until_blocked(state)
    return state


async def abort_experiment(experiment_id: str, reason: str, save_partial: bool) -> ResearchState:
    async def _op() -> ResearchState:
        return await _abort_experiment_unlocked(experiment_id, reason, save_partial)

    return await _with_experiment_lock(experiment_id, _op)


async def _abort_experiment_unlocked(experiment_id: str, reason: str, save_partial: bool) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    previous_status = _state_value(state, "status")
    previous_phase = _state_value(state, "phase")
    if _is_terminal(state):
        return state
    logger.warning("experiment.abort", experiment_id=experiment_id, reason=reason, save_partial=save_partial)

    state["status"] = ExperimentStatus.ABORTED.value
    state["phase"] = "aborted"
    state["timestamp_end"] = time.time()

    project = Path(state["project_path"])
    error_report = project / "docs" / "error_report.md"
    report_content = f"# Aborted\n\nReason: {reason}\n"
    state["documentation_path"] = str(error_report)
    state["documentation_content"] = report_content

    if not is_vscode_execution_mode(state):
        error_report.parent.mkdir(parents=True, exist_ok=True)
        error_report.write_text(report_content, encoding="utf-8")
        if save_partial:
            partial = project / "outputs" / "partial"
            partial.mkdir(parents=True, exist_ok=True)

    await ExperimentRepository.update(experiment_id, state)
    await _emit_state_webhook(
        state,
        "experiment.aborted",
        {"reason": reason, "save_partial": bool(save_partial)},
    )
    await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
    await ExperimentRepository.add_log(experiment_id, "system", "warning", "Experiment aborted", {"reason": reason})
    return state


async def retry_experiment(
    experiment_id: str,
    from_phase: str | None,
    reset_retries: bool,
    override_config: dict[str, Any] | None,
) -> ResearchState:
    async def _op() -> ResearchState:
        return await _retry_experiment_unlocked(
            experiment_id=experiment_id,
            from_phase=from_phase,
            reset_retries=reset_retries,
            override_config=override_config,
        )

    return await _with_experiment_lock(experiment_id, _op)


async def _retry_experiment_unlocked(
    experiment_id: str,
    from_phase: str | None,
    reset_retries: bool,
    override_config: dict[str, Any] | None,
) -> ResearchState:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")
    previous_status = _state_value(state, "status")
    previous_phase = _state_value(state, "phase")
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
    await _emit_state_webhook(
        state,
        "experiment.retried",
        {"from_phase": from_phase or "error_recovery", "reset_retries": bool(reset_retries)},
    )
    await _emit_transition_webhooks(state, previous_status=previous_status, previous_phase=previous_phase)
    await ExperimentRepository.add_log(experiment_id, "system", "info", "Experiment retry initiated")
    if settings.WORKFLOW_BACKGROUND_ENABLED:
        _schedule_background_run(experiment_id)
        return state
    return await run_until_blocked(state)


async def update_experiment_fields(
    experiment_id: str,
    updates: dict[str, Any],
    merge_nested: bool = True,
) -> tuple[ResearchState, dict[str, Any], dict[str, str]]:
    async def _op() -> tuple[ResearchState, dict[str, Any], dict[str, str]]:
        return await _update_experiment_fields_unlocked(experiment_id, updates, merge_nested)

    return await _with_experiment_lock(experiment_id, _op)


async def _update_experiment_fields_unlocked(
    experiment_id: str,
    updates: dict[str, Any],
    merge_nested: bool = True,
) -> tuple[ResearchState, dict[str, Any], dict[str, str]]:
    state = await ExperimentRepository.get(experiment_id)
    if not state:
        raise ValueError("Experiment not found")

    status_value = str(state.get("status") or "")
    if status_value == ExperimentStatus.RUNNING.value:
        raise RuntimeError("Experiment is running; wait until it is blocked before updating fields")

    applied: dict[str, Any] = {}
    rejected: dict[str, str] = {}
    candidate_prompt = str(state.get("user_prompt") or "")
    candidate_research_type = str(state.get("research_type") or "ai")
    for key, raw_value in (updates or {}).items():
        if key not in _PATCHABLE_FIELDS:
            rejected[key] = "field_not_patchable"
            continue

        if key in _PATCHABLE_STR_FIELDS:
            value = str(raw_value or "").strip()
            if key == "user_prompt":
                if len(value) < 10:
                    rejected[key] = "prompt_too_short"
                    continue
                value = value[:2000]
                try:
                    supported, _resolved, _reason = await validate_supported_prompt(
                        value,
                        research_type_hint=candidate_research_type,
                        experiment_id=experiment_id,
                        phase="api_patch_domain_gate",
                    )
                except Exception:
                    rejected[key] = "domain_classifier_unavailable"
                    continue
                if not supported:
                    rejected[key] = "unsupported_domain"
                    continue
                candidate_prompt = value
            elif key == "research_type":
                normalized = _normalize_patch_choice(value, {"ai", "quantum"})
                if normalized is None:
                    rejected[key] = "unsupported_value"
                    continue
                try:
                    supported, _resolved, _reason = await validate_supported_prompt(
                        candidate_prompt,
                        research_type_hint=normalized,
                        experiment_id=experiment_id,
                        phase="api_patch_domain_gate",
                    )
                except Exception:
                    rejected[key] = "domain_classifier_unavailable"
                    continue
                if not supported:
                    rejected[key] = "unsupported_domain"
                    continue
                value = normalized
                candidate_research_type = normalized
            elif key == "dataset_source":
                normalized = _normalize_patch_choice(value, {"synthetic", "sklearn", "upload", "kaggle"})
                if normalized is None:
                    rejected[key] = "unsupported_value"
                    continue
                value = normalized
            elif key == "output_format":
                normalized = _normalize_patch_choice(value, {".py", ".ipynb", "hybrid"})
                if normalized is None:
                    rejected[key] = "unsupported_value"
                    continue
                value = normalized
            elif key == "framework":
                value = value[:80]
            elif key in {"target_metric", "hardware_target", "kaggle_dataset_id", "quantum_framework", "quantum_algorithm", "quantum_backend"}:
                value = value[:120]
            state[key] = value
            applied[key] = value
            continue

        if key in _PATCHABLE_INT_FIELDS:
            try:
                value = int(raw_value)
            except Exception:
                rejected[key] = "expected_integer"
                continue
            if key == "max_epochs":
                value = max(1, min(value, 10000))
            elif key == "batch_size":
                value = max(1, min(value, 4096))
            elif key == "random_seed":
                value = max(0, min(value, 2_147_483_647))
            elif key == "quantum_qubit_count":
                value = max(1, min(value, 128))
            state[key] = value
            applied[key] = value
            continue

        if key in {"clarifications", "research_plan"}:
            if not isinstance(raw_value, dict):
                rejected[key] = "expected_object"
                continue
            if merge_nested:
                merged = dict(state.get(key) or {})
                merged.update(raw_value)
                state[key] = merged
            else:
                state[key] = dict(raw_value)
            applied[key] = state[key]
            continue

        if key == "requires_quantum":
            parsed = _bool_from_value(raw_value)
            if parsed is None:
                rejected[key] = "expected_boolean"
                continue
            state[key] = parsed
            applied[key] = parsed
            continue
        if key == "default_allow_research":
            parsed = _bool_from_value(raw_value)
            if parsed is None:
                rejected[key] = "expected_boolean"
                continue
            state[key] = parsed
            applied[key] = parsed
            continue

        rejected[key] = "unsupported_value"

    if not applied:
        return state, applied, rejected

    if "research_type" in applied:
        if str(state.get("research_type") or "ai").strip().lower() == "quantum":
            state["requires_quantum"] = True
            applied["requires_quantum"] = True
        elif "requires_quantum" not in applied:
            state["requires_quantum"] = False
            applied["requires_quantum"] = False

    state["total_duration_sec"] = max(0.0, time.time() - float(state.get("timestamp_start") or time.time()))
    await ExperimentRepository.update(experiment_id, state)
    await ExperimentRepository.add_log(
        experiment_id,
        "system",
        "info",
        "Experiment fields updated",
        {"applied_fields": sorted(applied.keys()), "rejected_fields": rejected},
    )
    if bool(applied.get("default_allow_research")) and settings.WORKFLOW_BACKGROUND_ENABLED and not _is_terminal(state):
        _schedule_background_run(experiment_id)
    return state, applied, rejected


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
    research_type = "quantum" if bool(state_row.get("requires_quantum")) else "ai"
    raw_state = state_row.get("state_json")
    if isinstance(raw_state, str) and raw_state.strip():
        try:
            parsed = json.loads(raw_state)
            value = str(parsed.get("research_type", research_type)).strip().lower()
            if value in {"ai", "quantum"}:
                research_type = value
        except Exception:
            pass
    return {
        "experiment_id": state_row["id"],
        "status": state_row["status"],
        "phase": state_row["phase"],
        "prompt_preview": (state_row["prompt"] or "")[:80],
        "research_type": research_type,
        "requires_quantum": bool(state_row["requires_quantum"]),
        "framework": state_row.get("framework"),
        "primary_metric": {"name": state_row.get("target_metric"), "value": None},
        "created_at": state_row.get("created_at"),
        "duration_sec": duration,
    }
