from __future__ import annotations

import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

from src.config.settings import settings
from src.core.logger import get_logger
from src.core.package_installer import dry_run_install, install_package, parse_package_spec
from src.state.research_state import DenialRecord, ExperimentStatus, ResearchState

logger = get_logger(__name__)

FALLBACK_MAP = {
    "torch": "scikit-learn==1.4.2",
    "tensorflow": "scikit-learn==1.4.2",
    "pennylane": "qiskit-aer==0.14.0",
    "qiskit": "pennylane==0.36.0",
    "kaggle": "requests==2.31.0",
    "jupyter": "__skip__",
}


def _fallback_for(package: str) -> str:
    for key, value in FALLBACK_MAP.items():
        if package.startswith(key):
            return value
    return ""


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


async def env_manager_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "env_manager"
    logger.info("agent.env_manager.start", experiment_id=state["experiment_id"])

    if settings.EXPERIMENT_VENV_ENABLED:
        venv_path = Path(state["venv_path"])
        python_path = venv_path / ("Scripts" if sys.platform.startswith("win") else "bin") / ("python.exe" if sys.platform.startswith("win") else "python")
        if not python_path.exists():
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
                logger.info("agent.env_manager.venv.created", experiment_id=state["experiment_id"], venv_path=str(venv_path))

    denied_items = {d["denied_item"] for d in state["denied_actions"]}
    for spec in state["required_packages"]:
        if spec in state["installed_packages"] or spec in denied_items:
            continue

        package, version = parse_package_spec(spec)
        if settings.AUTO_CONFIRM_LOW_RISK and _is_low_risk_package(package):
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
                    continue
            state["installed_packages"].append(spec)
            logger.info(
                "agent.env_manager.auto_confirmed_low_risk",
                experiment_id=state["experiment_id"],
                package=spec,
            )
            continue

        fallback = _fallback_for(package)
        state["pending_user_confirm"] = {
            "action_id": f"act_{uuid.uuid4().hex[:8]}",
            "action": "install_package",
            "package": package,
            "version": version,
            "pip_flags": ["--quiet", "--no-cache-dir"],
            "fallback_if_denied": fallback,
            "reason": f"Required for planned framework {state['framework']}",
            "dry_run": None,
        }

        # Optional dry-run validation.
        if settings.ENABLE_PACKAGE_INSTALL:
            result = dry_run_install(package, version, ["--quiet"])
            state["pending_user_confirm"]["dry_run"] = {"returncode": result.returncode, "stderr": result.stderr[-400:]}

        state["confirmations_requested"] = int(state.get("confirmations_requested", 0)) + 1
        state["status"] = ExperimentStatus.WAITING.value
        logger.info(
            "agent.env_manager.pending_confirmation",
            experiment_id=state["experiment_id"],
            package=package,
            version=version,
            fallback=fallback,
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
) -> ResearchState:
    logger.info("agent.env_manager.confirmation.start", experiment_id=state["experiment_id"], action_id=action_id, decision=decision)
    pending = state.get("pending_user_confirm")
    if not pending or pending.get("action_id") != action_id:
        raise ValueError("Unknown or stale action_id")

    spec = f"{pending['package']}=={pending['version']}" if pending.get("version") else pending["package"]

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
        if settings.ENABLE_PACKAGE_INSTALL:
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
                state["installed_packages"].append(spec)
        else:
            state["installed_packages"].append(spec)
        state["pending_user_confirm"] = None
        state["status"] = ExperimentStatus.RUNNING.value
        logger.info("agent.env_manager.confirmation.accepted", experiment_id=state["experiment_id"], package=spec)
        return state

    fallback = pending.get("fallback_if_denied", "")
    state["confirmations_processed"] = int(state.get("confirmations_processed", 0)) + 1
    denial: DenialRecord = {
        "action": "install_package",
        "denied_item": spec,
        "reason": reason,
        "alternative_offered": fallback,
        "alternative_accepted": False,
        "timestamp": time.time(),
    }
    state["denied_actions"].append(denial)

    if fallback and fallback != "__skip__":
        fb_package, fb_version = parse_package_spec(alternative_preference or fallback)
        state["pending_user_confirm"] = {
            "action_id": f"act_{uuid.uuid4().hex[:8]}",
            "action": "install_package",
            "package": fb_package,
            "version": fb_version,
            "pip_flags": ["--quiet", "--no-cache-dir"],
            "fallback_if_denied": "",
            "reason": f"Fallback for denied package {spec}",
        }
        state["confirmations_requested"] = int(state.get("confirmations_requested", 0)) + 1
        state["status"] = ExperimentStatus.WAITING.value
        logger.info("agent.env_manager.confirmation.fallback_offered", experiment_id=state["experiment_id"], fallback=fallback)
        return state

    state["pending_user_confirm"] = None
    state["status"] = ExperimentStatus.RUNNING.value
    logger.info("agent.env_manager.confirmation.denied_no_fallback", experiment_id=state["experiment_id"], denied=spec)
    return state
