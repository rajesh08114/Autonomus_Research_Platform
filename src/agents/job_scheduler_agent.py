from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode
from src.core.logger import get_logger
from src.core.user_behavior import build_user_behavior_profile
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ResearchState

logger = get_logger(__name__)


def _wants_notebook_artifacts(state: ResearchState) -> bool:
    fmt = str(state.get("output_format", ".py")).strip().lower()
    return fmt in {".ipynb", "hybrid"}


def _default_execution_order(state: ResearchState, project: Path) -> list[str]:
    code_level = str((state.get("research_plan") or {}).get("code_level", "intermediate")).strip().lower()
    algorithm_class = str((state.get("research_plan") or {}).get("algorithm_class", "supervised")).strip().lower()
    order: list[str] = []
    order.append(str(project / "data" / "validate_data.py"))
    if code_level == "advanced":
        order.append(str(project / "src" / "preprocessing.py"))
    if _wants_notebook_artifacts(state):
        order.append(str(project / "notebooks" / "run_notebook.py"))
    if algorithm_class in {"reinforcement", "quantum_ml"}:
        order.append(str(project / "main.py"))
    else:
        order.extend([str(project / "main.py")])
    deduped: list[str] = []
    for item in order:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _planned_or_created_files(state: ResearchState) -> set[str]:
    files = {str(Path(path).resolve()) for path in state.get("created_files", []) if str(path).strip()}
    for item in state.get("local_file_plan", []) or []:
        path = str((item or {}).get("path", "")).strip()
        if path:
            files.add(str(Path(path).resolve()))
    return files


def _latest_file_content_from_plan(state: ResearchState, absolute_path: str) -> str:
    target = str(Path(absolute_path).resolve())
    for item in reversed(state.get("local_file_plan", []) or []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        if str(Path(path).resolve()) != target:
            continue
        return str(item.get("content", ""))
    return ""


def _read_script_content(state: ResearchState, script_path: str, local_mode: bool) -> str:
    path = Path(script_path).resolve()
    if local_mode:
        from_plan = _latest_file_content_from_plan(state, str(path))
        if from_plan:
            return from_plan
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _notebook_syntax_violations(notebook_content: str, label: str) -> list[str]:
    violations: list[str] = []
    try:
        payload = json.loads(notebook_content)
    except Exception as exc:
        return [f"{label} invalid JSON: {exc}"]
    cells = payload.get("cells") if isinstance(payload, dict) else None
    if not isinstance(cells, list) or not cells:
        return [f"{label} has no cells"]
    for idx, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or str(cell.get("cell_type", "")) != "code":
            continue
        source = "".join(cell.get("source") or [])
        filtered = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            filtered.append(line)
        code = "\n".join(filtered).strip()
        if not code:
            continue
        try:
            ast.parse(code, filename=f"{label}::cell_{idx}")
        except SyntaxError as exc:
            violations.append(f"{label} cell {idx} syntax error: {exc.msg}")
    return violations


def _preflight_violations(state: ResearchState, candidates: list[str], local_mode: bool) -> list[str]:
    violations: list[str] = []
    project = Path(state["project_path"]).resolve()
    main_path = str((project / "main.py").resolve())
    notebook_mode = _wants_notebook_artifacts(state)
    notebook_runner_path = str((project / "notebooks" / "run_notebook.py").resolve())
    notebook_file_path = str((project / "notebooks" / "research_workflow.ipynb").resolve())
    if main_path not in candidates:
        violations.append("main.py missing from scheduler candidates")
    if notebook_mode and notebook_runner_path not in candidates:
        violations.append("notebooks/run_notebook.py missing from scheduler candidates in notebook/hybrid mode")

    report = state.get("data_report") if isinstance(state.get("data_report"), dict) else {}
    shape = report.get("shape", []) if isinstance(report, dict) else []
    if not isinstance(shape, list) or len(shape) < 2:
        violations.append("data_report.shape is invalid before scheduling")
    else:
        try:
            if int(shape[0]) <= 0:
                violations.append("dataset appears empty before scheduling")
        except Exception:
            violations.append("data_report.shape row count is non-numeric")

    for script in candidates:
        script_path = Path(script).resolve()
        if script_path.suffix.lower() != ".py":
            continue
        content = _read_script_content(state, str(script_path), local_mode)
        if not content.strip():
            violations.append(f"{script_path.name} has empty content")
            continue
        try:
            ast.parse(content, filename=str(script_path))
        except SyntaxError as exc:
            violations.append(f"{script_path.name} syntax error: {exc.msg}")

    if notebook_mode:
        notebook_content = _read_script_content(state, notebook_file_path, local_mode)
        if not notebook_content.strip():
            violations.append("notebooks/research_workflow.ipynb is missing or empty")
        else:
            violations.extend(_notebook_syntax_violations(notebook_content, "notebooks/research_workflow.ipynb"))
    return violations


async def _invoke_dynamic_scheduler(state: ResearchState, candidates: list[str]) -> dict[str, Any]:
    system_prompt = (
        "SYSTEM ROLE: job_scheduler_dynamic_plan.\n"
        "Return JSON only with keys:\n"
        "- execution_order: array of absolute script paths\n"
        "- rationale: short string\n"
        "Use only candidate scripts from context."
    )
    user_prompt = json.dumps(
        {
            "candidates": candidates,
            "research_plan": state.get("research_plan", {}),
            "requires_quantum": state.get("requires_quantum"),
            "framework": state.get("framework"),
            "dataset_source": state.get("dataset_source"),
            "current_phase": state.get("phase"),
            "user_behavior_profile": build_user_behavior_profile(state),
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="job_scheduler",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = parse_json_object(raw)
    if not parsed:
        logger.warning("agent.job_scheduler.dynamic_parse_failed", experiment_id=state["experiment_id"])
    return parsed


def _sanitize_dynamic_execution_order(
    *,
    payload: dict[str, Any],
    candidates: list[str],
    project: Path,
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    candidate_set = {str(Path(item).resolve()) for item in candidates}
    raw_order = payload.get("execution_order")
    if not isinstance(raw_order, list):
        return [], ["execution_order must be an array"]
    order: list[str] = []
    for idx, item in enumerate(raw_order):
        path_text = str(item or "").strip()
        if not path_text:
            violations.append(f"execution_order[{idx}] empty path")
            continue
        resolved = str(Path(path_text).resolve())
        if resolved not in candidate_set:
            violations.append(f"execution_order[{idx}] not in candidates: {resolved}")
            continue
        try:
            Path(resolved).resolve().relative_to(project.resolve())
        except ValueError:
            violations.append(f"execution_order[{idx}] outside project: {resolved}")
            continue
        if resolved not in order:
            order.append(resolved)
    return order, violations


async def job_scheduler_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "job_scheduler"
    logger.info("agent.scheduler.start", experiment_id=state["experiment_id"])
    project = Path(state["project_path"])
    local_mode = is_vscode_execution_mode(state)
    default_order = _default_execution_order(state, project)
    available_local_files = _planned_or_created_files(state)
    candidates = []
    for script in default_order:
        resolved = str(Path(script).resolve())
        if local_mode:
            if resolved in available_local_files:
                candidates.append(resolved)
        elif Path(resolved).exists():
            candidates.append(resolved)

    used_dynamic = False
    fallback_static = False
    execution_order = list(candidates)
    if candidates:
        payload = await _invoke_dynamic_scheduler(state, candidates)
        if not payload:
            if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                fallback_static = True
                logger.warning("agent.job_scheduler.dynamic_fallback_static", experiment_id=state["experiment_id"], reason="parse_failed")
            else:
                raise RuntimeError("Job scheduler dynamic parse failed")
        else:
            dynamic_order, violations = _sanitize_dynamic_execution_order(payload=payload, candidates=candidates, project=project)
            if violations:
                logger.warning("agent.job_scheduler.dynamic_validation_failed", experiment_id=state["experiment_id"], violations=violations)
                if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                    fallback_static = True
                    logger.warning(
                        "agent.job_scheduler.dynamic_fallback_static",
                        experiment_id=state["experiment_id"],
                        reason="validation_failed",
                    )
                else:
                    raise RuntimeError(f"Job scheduler dynamic validation failed: {violations}")
            else:
                execution_order = dynamic_order
                used_dynamic = True

    main_path = str((project / "main.py").resolve())
    if main_path in candidates and main_path not in execution_order:
        execution_order.append(main_path)
    notebook_runner_path = str((project / "notebooks" / "run_notebook.py").resolve())
    if _wants_notebook_artifacts(state) and notebook_runner_path in candidates and notebook_runner_path not in execution_order:
        insert_at = 1 if execution_order and execution_order[0].endswith("validate_data.py") else 0
        execution_order.insert(insert_at, notebook_runner_path)

    preflight = _preflight_violations(state, execution_order, local_mode=local_mode)
    if preflight:
        state.setdefault("research_plan", {})["scheduler_dynamic_plan_summary"] = {
            "used_dynamic": used_dynamic,
            "fallback_static": fallback_static,
            "candidate_count": len(candidates),
            "execution_order_count": len(execution_order),
            "preflight_violations": preflight[:10],
        }
        state["errors"].append(
            {
                "category": "SchedulerPreflightError",
                "message": "; ".join(preflight[:4])[:1000],
                "file_path": "job_scheduler",
                "line_number": 0,
                "traceback": "; ".join(preflight)[:2000],
                "timestamp": time.time(),
            }
        )
        state["phase"] = "error_recovery"
        logger.warning(
            "agent.scheduler.preflight_failed",
            experiment_id=state["experiment_id"],
            violations=preflight,
        )
        return state

    state["execution_order"] = execution_order
    state.setdefault("research_plan", {})["scheduler_dynamic_plan_summary"] = {
        "used_dynamic": used_dynamic,
        "fallback_static": fallback_static,
        "candidate_count": len(candidates),
        "execution_order_count": len(execution_order),
        "preflight_violations": [],
    }
    logger.info("agent.scheduler.end", experiment_id=state["experiment_id"], execution_order=state["execution_order"])
    return state
