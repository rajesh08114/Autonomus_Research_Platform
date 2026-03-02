from __future__ import annotations

import json
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode, local_python_command, local_python_for_state
from src.core.logger import get_logger
from src.core.package_installer import dry_run_install, install_package, parse_package_spec
from src.core.user_behavior import build_user_behavior_profile
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import DenialRecord, ExperimentStatus, ResearchState

logger = get_logger(__name__)


def _normalize_state_paths(state: ResearchState) -> None:
    project_path_raw = str(state.get("project_path") or "").strip()
    if not project_path_raw:
        return
    project_path = Path(project_path_raw).expanduser().resolve()
    desired_venv = (project_path / ".venv").resolve()
    current_venv_raw = str(state.get("venv_path") or "").strip()
    if not current_venv_raw:
        state["venv_path"] = str(desired_venv)
        return
    current_venv = Path(current_venv_raw).expanduser()
    # Repair legacy malformed values such as "...\\exp_123abc.venv"
    if current_venv.name != ".venv":
        state["venv_path"] = str(desired_venv)


def _pending_matches(
    state: ResearchState,
    *,
    action: str,
    phase: str,
    command: list[str] | None = None,
    package: str = "",
    version: str = "",
) -> bool:
    pending = state.get("pending_user_confirm") or {}
    if str(pending.get("action", "")) != action:
        return False
    if str(pending.get("phase", "")) != phase:
        return False
    if package and str(pending.get("package", "")).strip().lower() != package.strip().lower():
        return False
    if version and str(pending.get("version", "")).strip() != version.strip():
        return False
    if command is not None:
        existing_command = pending.get("command")
        if not isinstance(existing_command, list):
            return False
        normalized_existing = [str(item) for item in existing_command]
        normalized_target = [str(item) for item in command]
        if normalized_existing != normalized_target:
            return False
    return True


def _is_venv_interpreter_failure(stderr: str) -> bool:
    text = str(stderr or "").strip().lower()
    if not text:
        return False
    signals = (
        "no such file or directory",
        "cannot find the path specified",
        "is not recognized as an internal or external command",
        "file not found",
        "spawn",
        "enoent",
    )
    return any(signal in text for signal in signals)


def _normalized_spec_text(spec: str) -> str:
    package, version = parse_package_spec(str(spec or ""))
    package_norm = package.strip().lower()
    version_norm = version.strip()
    if not package_norm:
        return ""
    return f"{package_norm}=={version_norm}" if version_norm else package_norm


def _add_installed_spec(state: ResearchState, spec: str) -> None:
    normalized = _normalized_spec_text(spec)
    if not normalized:
        return
    existing = {_normalized_spec_text(item) for item in state.get("installed_packages", [])}
    if normalized not in existing:
        state["installed_packages"].append(normalized)


def _is_low_risk_package(package: str) -> bool:
    allowed = {p.strip().lower() for p in settings.LOW_RISK_PACKAGES.split(",") if p.strip()}
    return package.strip().lower() in allowed


def _should_inject_failure(point: str) -> bool:
    if not settings.FAILURE_INJECTION_ENABLED:
        return False
    configured = {p.strip().lower() for p in settings.FAILURE_INJECTION_POINTS.split(",") if p.strip()}
    if configured and point.lower() not in configured:
        return False
    return random.random() < float(max(0.0, min(1.0, settings.FAILURE_INJECTION_RATE)))


def _record_local_action(state: ResearchState, pending: dict[str, Any], decision: str, execution_result: dict[str, Any] | None) -> None:
    history = state.setdefault("local_action_history", [])
    history.append(
        {
            "action_id": pending.get("action_id"),
            "action": pending.get("action"),
            "phase": pending.get("phase"),
            "decision": decision,
            "timestamp": time.time(),
            "returncode": (execution_result or {}).get("returncode"),
        }
    )


def _normalized_exec_result(execution_result: dict[str, Any] | None) -> dict[str, Any]:
    payload = execution_result or {}
    command = payload.get("command")
    return {
        "returncode": int(payload.get("returncode", 0)),
        "stdout": str(payload.get("stdout", "")),
        "stderr": str(payload.get("stderr", "")),
        "duration_sec": float(payload.get("duration_sec", 0.0)),
        "command": [str(item) for item in command] if isinstance(command, list) else [],
        "cwd": str(payload.get("cwd", "")),
        "created_files": [str(item) for item in payload.get("created_files", [])] if isinstance(payload.get("created_files"), list) else [],
        "metadata": payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
    }


def _require_local_execution_result(local_mode: bool, decision: str, action: str, execution_result: dict[str, Any] | None) -> None:
    if local_mode and decision == "confirm" and action in {"prepare_venv", "install_package", "run_local_commands", "apply_file_operations"} and execution_result is None:
        raise RuntimeError("execution_result is required for local confirmation actions")


def _candidate_specs(state: ResearchState) -> list[str]:
    local_profile = state.get("local_hardware_profile") if isinstance(state.get("local_hardware_profile"), dict) else {}
    local_specs = {
        _normalized_spec_text(str(item))
        for item in (local_profile.get("python_packages") or [])
        if _normalized_spec_text(str(item))
    }
    installed_specs = {
        _normalized_spec_text(str(item))
        for item in state.get("installed_packages", [])
        if _normalized_spec_text(str(item))
    }
    denied_items = {
        _normalized_spec_text(str(d.get("denied_item", "")))
        for d in state.get("denied_actions", [])
        if _normalized_spec_text(str(d.get("denied_item", "")))
    }
    specs: list[str] = []
    for spec in state.get("required_packages") or []:
        spec_text = str(spec).strip()
        if not spec_text:
            continue
        spec_norm = _normalized_spec_text(spec_text)
        if spec_norm in local_specs:
            _add_installed_spec(state, spec_text)
            continue
        if spec_norm in installed_specs:
            continue
        if spec_norm in denied_items:
            continue
        specs.append(spec_norm)
    return specs


async def _invoke_env_dynamic_plan(state: ResearchState, candidates: list[str]) -> dict[str, Any]:
    system_prompt = (
        "SYSTEM ROLE: env_dynamic_next_action.\n"
        "Return JSON only with keys:\n"
        "- action: one of install_package or none\n"
        "- package_spec: exact pinned package from candidate list if action=install_package\n"
        "- reason: concise explanation\n"
        "Never suggest package_spec outside the candidate list."
    )
    user_prompt = json.dumps(
        {
            "required_packages": state.get("required_packages", []),
            "installed_packages": state.get("installed_packages", []),
            "denied_actions": state.get("denied_actions", []),
            "candidate_specs": candidates,
            "framework": state.get("framework"),
            "research_plan": state.get("research_plan", {}),
            "local_runtime": {
                "python_command": state.get("local_python_command"),
                "hardware_profile": state.get("local_hardware_profile", {}),
            },
            "user_behavior_profile": build_user_behavior_profile(state),
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="env_manager",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = parse_json_object(raw)
    if not parsed:
        logger.warning("agent.env.dynamic_parse_failed", experiment_id=state["experiment_id"])
    return parsed


def _validate_env_dynamic_plan(plan: dict[str, Any], candidates: list[str]) -> tuple[str, str, list[str]]:
    violations: list[str] = []
    action = str(plan.get("action", "")).strip().lower()
    if action not in {"install_package", "none"}:
        violations.append(f"unsupported action: {action}")
    package_spec = str(plan.get("package_spec", "")).strip()
    if action == "install_package" and package_spec not in candidates:
        violations.append(f"package_spec must be one of candidates: {package_spec}")
    if action == "none" and candidates:
        violations.append("action=none is invalid when candidate packages remain")
    return action, package_spec, violations


async def env_manager_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "env_manager"
    _normalize_state_paths(state)
    logger.info("agent.env_manager.start", experiment_id=state["experiment_id"])
    local_mode = is_vscode_execution_mode(state)
    base_local_python = str(state.get("local_python_command") or local_python_command()).strip() or local_python_command()
    local_python = local_python_for_state(state)
    hardware_profile = state.get("local_hardware_profile") if isinstance(state.get("local_hardware_profile"), dict) else {}
    gpu_name = str((hardware_profile or {}).get("gpu_name") or "").strip()

    if settings.EXPERIMENT_VENV_ENABLED:
        venv_path = Path(state["venv_path"])
        python_path = venv_path / ("Scripts" if sys.platform.startswith("win") else "bin") / ("python.exe" if sys.platform.startswith("win") else "python")
        if local_mode:
            # In VS Code execution mode, venv creation must happen on the user machine.
            if bool(state.get("venv_ready")):
                logger.info("agent.env_manager.venv.ready_local", experiment_id=state["experiment_id"], venv_path=str(venv_path))
            else:
                prepare_command = [base_local_python, "-m", "venv", state["venv_path"]]
                if _pending_matches(
                    state,
                    action="prepare_venv",
                    phase="env_manager",
                ):
                    state["status"] = ExperimentStatus.WAITING.value
                    return state
                state["pending_user_confirm"] = {
                    "action_id": f"act_{uuid.uuid4().hex[:8]}",
                    "action": "prepare_venv",
                    "phase": "env_manager",
                    "command": prepare_command,
                    "cwd": state["project_path"],
                    "reason": "Create an isolated virtual environment for this experiment",
                    "next_phase": "env_manager",
                }
                state["confirmations_requested"] = int(state.get("confirmations_requested", 0)) + 1
                state["status"] = ExperimentStatus.WAITING.value
                logger.info("agent.env_manager.pending_venv", experiment_id=state["experiment_id"], venv_path=str(venv_path))
                return state
        elif python_path.exists():
            state["venv_ready"] = True
        else:
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                state["errors"].append(
                    {
                        "category": "VenvCreateError",
                        "message": result.stderr[:1000],
                        "file_path": "env_manager",
                        "line_number": 0,
                        "traceback": result.stderr,
                        "timestamp": time.time(),
                    }
                )
            else:
                state["venv_ready"] = True
                logger.info("agent.env_manager.venv.created", experiment_id=state["experiment_id"], venv_path=str(venv_path))

    candidates = _candidate_specs(state)
    fallback_static = False
    used_dynamic = False
    selected_spec = candidates[0] if candidates else ""

    if candidates and settings.ENV_DYNAMIC_ENABLED:
        plan = await _invoke_env_dynamic_plan(state, candidates)
        if not plan:
            if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                fallback_static = True
                logger.warning("agent.env.dynamic_fallback_static", experiment_id=state["experiment_id"], reason="parse_failed")
            else:
                raise RuntimeError("Env dynamic planner failed: empty/invalid LLM JSON response")
        else:
            action, package_spec, violations = _validate_env_dynamic_plan(plan, candidates)
            if violations:
                logger.warning("agent.env.dynamic_validation_failed", experiment_id=state["experiment_id"], violations=violations)
                if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                    fallback_static = True
                    logger.warning("agent.env.dynamic_fallback_static", experiment_id=state["experiment_id"], reason="validation_failed")
                else:
                    raise RuntimeError(f"Env dynamic planner validation failed: {violations}")
            elif action == "install_package":
                selected_spec = package_spec
                used_dynamic = True
            else:
                if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                    fallback_static = True
                    logger.warning("agent.env.dynamic_fallback_static", experiment_id=state["experiment_id"], reason="none_action")
                else:
                    raise RuntimeError("Env dynamic planner returned action=none while package candidates remain")

    state.setdefault("research_plan", {})["env_dynamic_plan_summary"] = {
        "enabled": bool(settings.ENV_DYNAMIC_ENABLED),
        "used_dynamic": used_dynamic,
        "fallback_static": fallback_static,
        "candidate_count": len(candidates),
        "selected_package": selected_spec,
        "local_python_command": local_python,
        "base_local_python_command": base_local_python,
        "hardware_target": state.get("hardware_target", "cpu"),
        "gpu_name": gpu_name or None,
    }

    if selected_spec:
        package, version = parse_package_spec(selected_spec)
        if not local_mode and settings.AUTO_CONFIRM_LOW_RISK and _is_low_risk_package(package):
            if settings.ENABLE_PACKAGE_INSTALL:
                result = install_package(package, version, ["--quiet", "--no-cache-dir"])
                if result.returncode != 0:
                    state["errors"].append(
                        {
                            "category": "InstallError",
                            "message": result.stderr[:1000],
                            "file_path": "env_manager",
                            "line_number": 0,
                            "traceback": result.stderr,
                            "timestamp": time.time(),
                        }
                    )
                else:
                    _add_installed_spec(state, selected_spec)
            else:
                _add_installed_spec(state, selected_spec)
            logger.info(
                "agent.env_manager.auto_confirmed_low_risk",
                experiment_id=state["experiment_id"],
                package=selected_spec,
            )
        else:
            install_command = [local_python, "-m", "pip", "install", selected_spec, "--quiet", "--no-cache-dir"]
            if _pending_matches(
                state,
                action="install_package",
                phase="env_manager",
                package=package,
                version=version,
            ):
                state["status"] = ExperimentStatus.WAITING.value
                return state
            state["pending_user_confirm"] = {
                "action_id": f"act_{uuid.uuid4().hex[:8]}",
                "action": "install_package",
                "phase": "env_manager",
                "package": package,
                "version": version,
                "pip_flags": ["--quiet", "--no-cache-dir"],
                "command": install_command,
                "cwd": state["project_path"],
                "reason": (
                    f"Required for planned framework {state['framework']}"
                    + (f" on GPU {gpu_name}" if gpu_name else "")
                ),
                "dry_run": None,
                "next_phase": "env_manager",
            }

            if not local_mode and settings.ENABLE_PACKAGE_INSTALL:
                result = dry_run_install(package, version, ["--quiet"])
                state["pending_user_confirm"]["dry_run"] = {"returncode": result.returncode, "stderr": result.stderr[-400:]}

            state["confirmations_requested"] = int(state.get("confirmations_requested", 0)) + 1
            state["status"] = ExperimentStatus.WAITING.value
            logger.info(
                "agent.env_manager.pending_confirmation",
                experiment_id=state["experiment_id"],
                package=package,
                version=version,
            )
            return state

    state["pending_user_confirm"] = None
    state["status"] = ExperimentStatus.RUNNING.value
    logger.info("agent.env_manager.end", experiment_id=state["experiment_id"], installed_count=len(state["installed_packages"]))
    return state


async def apply_user_confirmation(
    state: ResearchState,
    action_id: str,
    decision: str,
    reason: str = "",
    alternative_preference: str = "",
    execution_result: dict[str, Any] | None = None,
) -> ResearchState:
    _ = alternative_preference
    logger.info("agent.env_manager.confirmation.start", experiment_id=state["experiment_id"], action_id=action_id, decision=decision)
    pending = state.get("pending_user_confirm")
    if not pending or pending.get("action_id") != action_id:
        raise ValueError("Unknown or stale action_id")

    local_mode = is_vscode_execution_mode(state)
    action = str(pending.get("action", ""))
    _require_local_execution_result(local_mode, decision, action, execution_result)
    result = _normalized_exec_result(execution_result)
    state["last_local_action_result"] = result
    _record_local_action(state, pending, decision, result)

    if action == "run_local_commands":
        state["confirmations_processed"] = int(state.get("confirmations_processed", 0)) + 1
        if decision == "confirm":
            command = result["command"] or [str(c) for c in pending.get("commands", [])]
            cwd = result["cwd"] or str(pending.get("cwd", state["project_path"]))
            script_path = str(pending.get("script_path", state.get("current_script") or "main.py"))
            log = {
                "script_path": script_path,
                "command": command,
                "cwd": cwd,
                "returncode": int(result["returncode"]),
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "duration_sec": float(result["duration_sec"]),
                "timestamp": time.time(),
                "executor": command[0] if command else local_python_command(),
                "host": "user_local_machine",
            }
            state["execution_logs"].append(log)
            for created in [str(item) for item in pending.get("created_files", [])]:
                if created not in state["created_files"]:
                    state["created_files"].append(created)
                if created not in state.setdefault("local_materialized_files", []):
                    state["local_materialized_files"].append(created)
            for created in result["created_files"]:
                if created not in state["created_files"]:
                    state["created_files"].append(created)
                if created not in state.setdefault("local_materialized_files", []):
                    state["local_materialized_files"].append(created)
            if log["returncode"] != 0:
                state["errors"].append(
                    {
                        "category": "LocalExecutionError",
                        "message": log["stderr"][:1000],
                        "file_path": script_path,
                        "line_number": 0,
                        "traceback": log["stderr"][:2000],
                        "timestamp": time.time(),
                    }
                )
            state["pending_user_confirm"] = None
            state["status"] = ExperimentStatus.RUNNING.value
            return state

        state["status"] = ExperimentStatus.ABORTED.value
        state["timestamp_end"] = time.time()
        state["pending_user_confirm"] = None
        return state

    if action == "apply_file_operations":
        state["confirmations_processed"] = int(state.get("confirmations_processed", 0)) + 1
        if decision == "confirm":
            for created in [str(item) for item in pending.get("created_files", [])]:
                if created not in state["created_files"]:
                    state["created_files"].append(created)
                if created not in state.setdefault("local_materialized_files", []):
                    state["local_materialized_files"].append(created)
            for created in result["created_files"]:
                if created not in state["created_files"]:
                    state["created_files"].append(created)
                if created not in state.setdefault("local_materialized_files", []):
                    state["local_materialized_files"].append(created)
            if result["returncode"] != 0:
                state["errors"].append(
                    {
                        "category": "LocalPhaseActionError",
                        "message": result["stderr"][:1000],
                        "file_path": str((pending.get("created_files") or ["phase_action"])[0]),
                        "line_number": 0,
                        "traceback": result["stderr"][:2000],
                        "timestamp": time.time(),
                    }
                )
                state["phase"] = "error_recovery"
            else:
                state["phase"] = str(pending.get("next_phase") or state.get("phase") or "planner")
            state["pending_user_confirm"] = None
            state["status"] = ExperimentStatus.RUNNING.value
            return state
        state["status"] = ExperimentStatus.ABORTED.value
        state["timestamp_end"] = time.time()
        state["pending_user_confirm"] = None
        return state

    if action == "prepare_venv":
        state["confirmations_processed"] = int(state.get("confirmations_processed", 0)) + 1
        if decision == "confirm":
            if result["returncode"] == 0:
                state["venv_ready"] = True
            else:
                state["errors"].append(
                    {
                        "category": "VenvCreateError",
                        "message": result["stderr"][:1000],
                        "file_path": "env_manager",
                        "line_number": 0,
                        "traceback": result["stderr"][:2000],
                        "timestamp": time.time(),
                    }
                )
            state["pending_user_confirm"] = None
            state["status"] = ExperimentStatus.RUNNING.value
            return state
        state["pending_user_confirm"] = None
        state["status"] = ExperimentStatus.RUNNING.value
        return state

    if action != "install_package":
        state["pending_user_confirm"] = None
        state["status"] = ExperimentStatus.RUNNING.value
        return state

    spec = _normalized_spec_text(
        f"{pending['package']}=={pending['version']}" if pending.get("version") else pending["package"]
    )

    if decision == "confirm":
        if _should_inject_failure("dependency_install"):
            state["confirmations_processed"] = int(state.get("confirmations_processed", 0)) + 1
            state["errors"].append(
                {
                    "category": "InstallError",
                    "message": "Injected dependency installation failure",
                    "file_path": "env_manager",
                    "line_number": 0,
                    "traceback": "Injected dependency installation failure",
                    "timestamp": time.time(),
                }
            )
            state["pending_user_confirm"] = None
            state["status"] = ExperimentStatus.RUNNING.value
            return state
        state["confirmations_processed"] = int(state.get("confirmations_processed", 0)) + 1
        if local_mode:
            if result["returncode"] != 0:
                state["errors"].append(
                    {
                        "category": "InstallError",
                        "message": result["stderr"][:1000],
                        "file_path": "env_manager",
                        "line_number": 0,
                        "traceback": result["stderr"][:2000],
                        "timestamp": time.time(),
                    }
                )
                state["denied_actions"].append(
                    {
                        "action": "install_package",
                        "denied_item": spec,
                        "reason": f"Local install failed (returncode={result['returncode']})",
                        "alternative_offered": "",
                        "alternative_accepted": False,
                        "timestamp": time.time(),
                    }
                )
                pending_command = pending.get("command")
                first_token = ""
                if isinstance(pending_command, list) and pending_command:
                    first_token = str(pending_command[0] or "")
                if ".venv" in first_token.replace("/", "\\").lower() and _is_venv_interpreter_failure(result["stderr"]):
                    # Force re-prepare of venv in the next env_manager cycle.
                    state["venv_ready"] = False
            else:
                metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                resolved_spec = _normalized_spec_text(str((metadata or {}).get("resolved_package_spec", "")))
                _add_installed_spec(state, resolved_spec or spec)
        elif settings.ENABLE_PACKAGE_INSTALL:
            result = install_package(pending["package"], pending.get("version", ""), pending.get("pip_flags", []))
            if result.returncode != 0:
                state["errors"].append(
                    {
                        "category": "InstallError",
                        "message": result.stderr[:1000],
                        "file_path": "env_manager",
                        "line_number": 0,
                        "traceback": result.stderr,
                        "timestamp": time.time(),
                    }
                )
            else:
                _add_installed_spec(state, spec)
        else:
            _add_installed_spec(state, spec)
        state["pending_user_confirm"] = None
        state["status"] = ExperimentStatus.RUNNING.value
        logger.info("agent.env_manager.confirmation.accepted", experiment_id=state["experiment_id"], package=spec)
        return state

    state["confirmations_processed"] = int(state.get("confirmations_processed", 0)) + 1
    denial: DenialRecord = {
        "action": "install_package",
        "denied_item": spec,
        "reason": reason,
        "alternative_offered": "",
        "alternative_accepted": False,
        "timestamp": time.time(),
    }
    state["denied_actions"].append(denial)

    state["pending_user_confirm"] = None
    state["status"] = ExperimentStatus.RUNNING.value
    logger.info("agent.env_manager.confirmation.denied", experiment_id=state["experiment_id"], denied=spec)
    return state
