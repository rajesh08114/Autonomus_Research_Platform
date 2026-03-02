from __future__ import annotations

import re
import time
from pathlib import Path

from src.config.settings import settings
from src.core.file_manager import read_text_file, replace_in_file, write_text_file
from src.core.logger import get_logger
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)


def _extract_trace_files(traceback_text: str, project_path: str) -> list[str]:
    project_norm = str(project_path).replace("/", "\\").lower()
    found = re.findall(r'File "([^"]+)"', str(traceback_text or ""))
    output: list[str] = []
    for raw in found:
        path = str(raw).strip()
        if not path:
            continue
        norm = path.replace("/", "\\").lower()
        if project_norm and project_norm not in norm:
            continue
        if path not in output:
            output.append(path)
    return output


def _inject_logger_if_missing(state: ResearchState, file_path: str) -> bool:
    try:
        content = read_text_file(state["project_path"], file_path)
    except Exception:
        return False
    text = str(content or "")
    if "logger." not in text:
        return False
    if re.search(r"^\s*logger\s*=", text, flags=re.MULTILINE):
        return False

    lines = text.splitlines()
    insert_idx = 0
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    while insert_idx < len(lines) and lines[insert_idx].startswith("from __future__ import"):
        insert_idx += 1

    has_logging_import = bool(re.search(r"^\s*import\s+logging\s*$", text, flags=re.MULTILINE))
    patch_lines: list[str] = []
    if not has_logging_import:
        patch_lines.append("import logging")
    patch_lines.extend(
        [
            "logger = logging.getLogger(__name__)",
            "if not logging.getLogger().handlers:",
            "    logging.basicConfig(level=logging.INFO)",
            "",
        ]
    )

    patched = lines[:insert_idx] + patch_lines + lines[insert_idx:]
    write_text_file(state["project_path"], file_path, "\n".join(patched).rstrip() + "\n")
    return True

async def error_recovery_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "error_recovery"
    logger.warning("agent.error_recovery.start", experiment_id=state["experiment_id"], retry_count=state["retry_count"])
    if state["retry_count"] >= settings.MAX_RETRY_COUNT:
        state["status"] = ExperimentStatus.ABORTED.value
        state["timestamp_end"] = time.time()
        return state

    if not state["errors"]:
        return state

    latest = state["errors"][-1]
    category = latest["category"]
    latest_message = str(latest.get("message") or "")
    latest_message_lower = latest_message.strip().lower()
    logger.warning("agent.error_recovery.classified", experiment_id=state["experiment_id"], category=category)
    state["retry_count"] += 1
    state["last_error_category"] = category

    if category in {"LocalExecutionError", "NameError", "unknown", "SchedulerPreflightError"}:
        trace_text = str(latest.get("traceback") or latest_message)
        trace_lower = trace_text.lower()
        if "nameerror" in trace_lower and "logger" in trace_lower:
            candidates = _extract_trace_files(trace_text, state["project_path"])
            patched_file = ""
            for candidate in reversed(candidates):
                if _inject_logger_if_missing(state, candidate):
                    patched_file = candidate
                    break
            if patched_file:
                state["repair_history"].append(
                    {
                        "attempt": state["retry_count"],
                        "error_category": category,
                        "fix_description": "Injected module logger bootstrap for NameError",
                        "file_changed": patched_file,
                        "find_text": "NameError: name 'logger' is not defined",
                        "replace_text": "import logging + logger initialization",
                        "timestamp": time.time(),
                    }
                )
                state["phase"] = "subprocess_runner"
                logger.info(
                    "agent.error_recovery.fix_logger_nameerror",
                    experiment_id=state["experiment_id"],
                    file=patched_file,
                )
                return state

        # If local execution failed and no direct patch was possible, regenerate
        # source with lower complexity to avoid infinite rerun loops.
        plan = state.setdefault("research_plan", {})
        old_level = str(plan.get("code_level", "intermediate")).strip().lower()
        next_level = "intermediate" if old_level == "advanced" else "low"
        plan["code_level"] = next_level
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": f"Runtime local execution failure; requested code regeneration ({old_level} -> {next_level})",
                "file_changed": "research_plan",
                "find_text": latest_message[:160],
                "replace_text": f"code_level={next_level}",
                "timestamp": time.time(),
            }
        )
        state["phase"] = "code_generator"
        logger.info(
            "agent.error_recovery.fix_regenerate_on_local_execution_error",
            experiment_id=state["experiment_id"],
            from_level=old_level,
            to_level=next_level,
            category=category,
        )
        return state

    if category in {"CodeGenerationStrictError", "PhaseExecutionError", "RuntimeError"} and (
        "strict_state_only" in latest_message_lower or "dynamic code generation failed" in latest_message_lower
    ):
        plan = state.setdefault("research_plan", {})
        clar = state.setdefault("clarifications", {})
        old_level = str(plan.get("code_level", clar.get("code_level", "intermediate"))).strip().lower()
        next_level = "intermediate" if old_level == "advanced" else "low"
        plan["code_level"] = next_level
        clar["code_level"] = next_level

        allowed_problem_types = {"classification", "regression", "clustering", "reinforcement", "forecasting", "generation"}
        allowed_algorithm_classes = {"supervised", "unsupervised", "reinforcement", "quantum_ml"}
        problem_type = str(plan.get("problem_type", clar.get("problem_type", "classification"))).strip().lower()
        algorithm_class = str(plan.get("algorithm_class", clar.get("algorithm_class", "supervised"))).strip().lower()
        if problem_type not in allowed_problem_types:
            problem_type = "classification"
        if algorithm_class not in allowed_algorithm_classes:
            algorithm_class = "supervised"
        plan["problem_type"] = problem_type
        clar["problem_type"] = problem_type
        plan["algorithm_class"] = algorithm_class
        clar["algorithm_class"] = algorithm_class

        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": (
                    f"Normalized strict codegen contract and lowered code complexity from {old_level} to {next_level}"
                ),
                "file_changed": "research_plan+clarifications",
                "find_text": latest_message[:120],
                "replace_text": f"problem_type={problem_type}, algorithm_class={algorithm_class}, code_level={next_level}",
                "timestamp": time.time(),
            }
        )
        state["phase"] = "code_generator"
        logger.info(
            "agent.error_recovery.fix_codegen_contract",
            experiment_id=state["experiment_id"],
            from_level=old_level,
            to_level=next_level,
            problem_type=problem_type,
            algorithm_class=algorithm_class,
        )
        return state

    if category in {"ModuleNotFoundError", "ImportError"}:
        # Let env manager resolve missing dependency on next pass.
        missing = "scikit-learn==1.5.2"
        if missing not in state["required_packages"]:
            state["required_packages"].append(missing)
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": "Added required package to required_packages",
                "file_changed": "requirements",
                "find_text": "",
                "replace_text": missing,
                "timestamp": time.time(),
            }
        )
        state["phase"] = "env_manager"
        logger.info("agent.error_recovery.fix_added_package", experiment_id=state["experiment_id"], package=missing)
        return state

    if category in {"gpu_unavailable", "RuntimeError"}:
        config_path = str(Path(state["project_path"]) / "config.py")
        ok = replace_in_file(state["project_path"], config_path, 'DEVICE = "cpu" if "', 'DEVICE = "cpu" if "', backup=True)
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": "Forced CPU execution",
                "file_changed": config_path,
                "find_text": "DEVICE",
                "replace_text": "cpu",
                "timestamp": time.time(),
            }
        )
        if not ok:
            state["status"] = ExperimentStatus.ABORTED.value
            state["timestamp_end"] = time.time()
            return state
        logger.info("agent.error_recovery.fix_cpu", experiment_id=state["experiment_id"], file=config_path)

    if category in {"SyntaxError", "IndentationError"}:
        plan = state.setdefault("research_plan", {})
        old_level = str(plan.get("code_level", "intermediate")).strip().lower()
        next_level = "intermediate" if old_level == "advanced" else "low"
        plan["code_level"] = next_level
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": f"Downgraded code generation complexity from {old_level} to {next_level}",
                "file_changed": "research_plan",
                "find_text": "code_level",
                "replace_text": next_level,
                "timestamp": time.time(),
            }
        )
        state["phase"] = "code_generator"
        logger.info("agent.error_recovery.fix_codegen_level", experiment_id=state["experiment_id"], from_level=old_level, to_level=next_level)
        return state

    if category in {"quantum_backend_error", "DeviceError"} and state.get("requires_quantum"):
        state["quantum_backend"] = "default.qubit"
        state["repair_history"].append(
            {
                "attempt": state["retry_count"],
                "error_category": category,
                "fix_description": "Switched quantum backend to default.qubit",
                "file_changed": "state.quantum_backend",
                "find_text": str(state.get("quantum_backend")),
                "replace_text": "default.qubit",
                "timestamp": time.time(),
            }
        )
        state["phase"] = "quantum_gate"
        logger.info("agent.error_recovery.fix_quantum_backend", experiment_id=state["experiment_id"], backend="default.qubit")
        return state

    if state["retry_count"] >= settings.MAX_RETRY_COUNT:
        state["status"] = ExperimentStatus.ABORTED.value
        state["timestamp_end"] = time.time()
        logger.error("agent.error_recovery.abort", experiment_id=state["experiment_id"], retry_count=state["retry_count"])
    return state
