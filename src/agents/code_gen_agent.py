from __future__ import annotations

import ast
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode
from src.core.file_manager import write_text_file
from src.core.local_actions import queue_local_file_action
from src.core.logger import get_logger
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ResearchState

logger = get_logger(__name__)

_ALLOWED_PROBLEM_TYPES = {"classification", "regression", "clustering", "reinforcement", "forecasting", "generation"}
_ALLOWED_CODE_LEVELS = {"low", "intermediate", "advanced"}
_ALLOWED_ALGORITHM_CLASSES = {"supervised", "unsupervised", "reinforcement", "quantum_ml"}
_REQUIRED_RELATIVE_FILES = (
    "config.py",
    "main.py",
    "src/__init__.py",
    "src/utils.py",
    "src/preprocessing.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
)
_FRAMEWORK_IMPORT_ROOTS: dict[str, set[str]] = {
    "sklearn": {"sklearn"},
    "scikit-learn": {"sklearn"},
    "pytorch": {"torch", "torchvision", "torchaudio"},
    "torch": {"torch", "torchvision", "torchaudio"},
    "tensorflow": {"tensorflow", "keras"},
    "xgboost": {"xgboost"},
    "lightgbm": {"lightgbm"},
    "catboost": {"catboost"},
    "qiskit": {"qiskit"},
    "pennylane": {"pennylane"},
}


def _should_inject_failure(point: str) -> bool:
    if not settings.FAILURE_INJECTION_ENABLED:
        return False
    configured = {p.strip().lower() for p in settings.FAILURE_INJECTION_POINTS.split(",") if p.strip()}
    if configured and point.lower() not in configured:
        return False
    return random.random() < float(max(0.0, min(1.0, settings.FAILURE_INJECTION_RATE)))


def _pick_state_value(*values: Any, default: str) -> str:
    for value in values:
        text = str(value or "").strip().lower()
        if text:
            return text
    return default


def _resolve_problem_type(state: ResearchState) -> str:
    selected = _pick_state_value(
        (state.get("research_plan") or {}).get("problem_type"),
        (state.get("clarifications") or {}).get("problem_type"),
        default="classification",
    )
    return selected if selected in _ALLOWED_PROBLEM_TYPES else "classification"


def _resolve_code_level(state: ResearchState) -> str:
    selected = _pick_state_value(
        (state.get("research_plan") or {}).get("code_level"),
        (state.get("clarifications") or {}).get("code_level"),
        default="intermediate",
    )
    return selected if selected in _ALLOWED_CODE_LEVELS else "intermediate"


def _resolve_algorithm_class(state: ResearchState) -> str:
    selected = _pick_state_value(
        (state.get("research_plan") or {}).get("algorithm_class"),
        (state.get("clarifications") or {}).get("algorithm_class"),
        default="supervised",
    )
    return selected if selected in _ALLOWED_ALGORITHM_CLASSES else "supervised"


def _extract_file_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("files", "source_files", "file_operations"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _to_project_path(project: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip())
    resolved = (candidate if candidate.is_absolute() else project / candidate).resolve()
    project_resolved = project.resolve()
    try:
        resolved.relative_to(project_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Generated file path is outside project root: {raw_path}") from exc
    return resolved


def _normalize_manifest(project: Path, items: list[dict[str, Any]]) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for item in items:
        raw_path = str(item.get("path") or item.get("file_path") or "").strip()
        if not raw_path:
            continue
        content = item.get("content")
        if content is None:
            content = item.get("code", "")
        abs_path = _to_project_path(project, raw_path)
        manifest[str(abs_path)] = str(content)
    return manifest


def _required_absolute_paths(project: Path) -> list[str]:
    return [str((project / rel).resolve()) for rel in _REQUIRED_RELATIVE_FILES]


def _ensure_config_contract(
    manifest: dict[str, str],
    project: Path,
    problem_type: str,
    code_level: str,
    algorithm_class: str,
    target_metric: str,
    hardware_target: str,
) -> None:
    config_path = str((project / "config.py").resolve())
    current = str(manifest.get(config_path, ""))
    patch_lines: list[str] = []
    if "PROBLEM_TYPE" not in current:
        patch_lines.append(f'PROBLEM_TYPE = "{problem_type}"')
    if "CODE_LEVEL" not in current:
        patch_lines.append(f'CODE_LEVEL = "{code_level}"')
    if "ALGORITHM_CLASS" not in current:
        patch_lines.append(f'ALGORITHM_CLASS = "{algorithm_class}"')
    if "TARGET_METRIC" not in current:
        patch_lines.append(f'TARGET_METRIC = "{target_metric}"')
    if "DEVICE" not in current:
        patch_lines.append(f'DEVICE = "cpu" if "{hardware_target}" != "cuda" else "cuda"')
    if patch_lines:
        suffix = "\n".join(patch_lines)
        manifest[config_path] = f"{current.rstrip()}\n\n{suffix}\n".lstrip()


def _validate_required_files(manifest: dict[str, str], project: Path) -> list[str]:
    missing: list[str] = []
    for path in _required_absolute_paths(project):
        if path not in manifest:
            missing.append(path)
            continue
        rel = str(Path(path).resolve().relative_to(project.resolve())).replace("\\", "/")
        if rel == "src/__init__.py":
            continue
        if not str(manifest.get(path, "")).strip():
            missing.append(path)
    return missing


def _normalize_import_root(spec: str) -> str:
    raw = str(spec or "").strip().lower()
    if not raw:
        return ""
    raw = raw.split("==", 1)[0]
    raw = raw.split("[", 1)[0]
    raw = raw.replace("-", "_")
    return raw.split(".", 1)[0]


def _build_allowed_import_roots(state: ResearchState) -> set[str]:
    allowed: set[str] = set(getattr(sys, "stdlib_module_names", set()))
    allowed.update({"config", "src"})
    for spec in state.get("required_packages") or []:
        root = _normalize_import_root(str(spec))
        if root:
            allowed.add(root)
    framework = str(state.get("framework") or "").strip().lower()
    allowed.update(_FRAMEWORK_IMPORT_ROOTS.get(framework, set()))
    if bool(state.get("requires_quantum")):
        q_framework = str(state.get("quantum_framework") or "").strip().lower()
        allowed.update(_FRAMEWORK_IMPORT_ROOTS.get(q_framework, set()))
    return allowed


def _collect_import_roots(content: str, file_path: str) -> tuple[set[str], list[str]]:
    roots: set[str] = set()
    errors: list[str] = []
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as exc:
        errors.append(f"{file_path}: syntax error during strict validation ({exc.msg})")
        return roots, errors
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name or "").strip()
                root = name.split(".", 1)[0]
                if root:
                    roots.add(root)
        elif isinstance(node, ast.ImportFrom):
            if int(getattr(node, "level", 0) or 0) > 0:
                continue
            module = str(getattr(node, "module", "") or "").strip()
            root = module.split(".", 1)[0] if module else ""
            if root:
                roots.add(root)
    return roots, errors


def _payload_contract_from_response(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "problem_type": str(payload.get("problem_type") or "").strip().lower(),
        "code_level": str(payload.get("code_level") or "").strip().lower(),
        "algorithm_class": str(payload.get("algorithm_class") or "").strip().lower(),
    }


def _extract_py_constant(content: str, name: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return ""
    return str(match.group(1)).strip().lower()


def _collect_strict_violations(
    *,
    state: ResearchState,
    payload: dict[str, Any],
    manifest: dict[str, str],
    project: Path,
    expected_problem_type: str,
    expected_code_level: str,
    expected_algorithm_class: str,
) -> list[str]:
    violations: list[str] = []
    missing_after_parse = _validate_required_files(manifest, project)
    if missing_after_parse:
        missing_rel = [str(Path(path).resolve().relative_to(project.resolve())).replace("\\", "/") for path in missing_after_parse]
        violations.append(f"Missing required files: {missing_rel}")

    if not settings.STRICT_STATE_ONLY:
        return violations

    contract = _payload_contract_from_response(payload)
    if contract["problem_type"] != expected_problem_type:
        violations.append(
            f"Payload problem_type mismatch: expected '{expected_problem_type}', got '{contract['problem_type'] or '<missing>'}'"
        )
    if contract["code_level"] != expected_code_level:
        violations.append(f"Payload code_level mismatch: expected '{expected_code_level}', got '{contract['code_level'] or '<missing>'}'")
    if settings.STRICT_STATE_ONLY_ENFORCE_ALGO_CLASS and contract["algorithm_class"] != expected_algorithm_class:
        violations.append(
            "Payload algorithm_class mismatch: "
            f"expected '{expected_algorithm_class}', got '{contract['algorithm_class'] or '<missing>'}'"
        )

    config_path = str((project / "config.py").resolve())
    config_content = str(manifest.get(config_path, ""))
    config_problem = _extract_py_constant(config_content, "PROBLEM_TYPE")
    config_code_level = _extract_py_constant(config_content, "CODE_LEVEL")
    config_algo = _extract_py_constant(config_content, "ALGORITHM_CLASS")
    if config_problem and config_problem != expected_problem_type:
        violations.append(
            f"config.py PROBLEM_TYPE mismatch: expected '{expected_problem_type}', got '{config_problem}'"
        )
    if config_code_level and config_code_level != expected_code_level:
        violations.append(f"config.py CODE_LEVEL mismatch: expected '{expected_code_level}', got '{config_code_level}'")
    if settings.STRICT_STATE_ONLY_ENFORCE_ALGO_CLASS and config_algo and config_algo != expected_algorithm_class:
        violations.append(
            f"config.py ALGORITHM_CLASS mismatch: expected '{expected_algorithm_class}', got '{config_algo}'"
        )

    if settings.STRICT_STATE_ONLY_ENFORCE_IMPORTS:
        allowed_roots = _build_allowed_import_roots(state)
        for path, content in manifest.items():
            if Path(path).suffix != ".py":
                continue
            roots, parse_errors = _collect_import_roots(content, path)
            violations.extend(parse_errors)
            disallowed = sorted(root for root in roots if root not in allowed_roots)
            if disallowed:
                rel = str(Path(path).resolve().relative_to(project.resolve())).replace("\\", "/")
                violations.append(f"{rel} uses disallowed import roots: {disallowed}")
    return violations


def _build_codegen_context(
    state: ResearchState,
    problem_type: str,
    code_level: str,
    algorithm_class: str,
) -> dict[str, Any]:
    return {
        "experiment_id": state.get("experiment_id"),
        "user_prompt": state.get("user_prompt"),
        "research_type": state.get("research_type"),
        "problem_type": problem_type,
        "code_level": code_level,
        "algorithm_class": algorithm_class,
        "framework": state.get("framework"),
        "python_version": state.get("python_version"),
        "target_metric": state.get("target_metric"),
        "requires_quantum": bool(state.get("requires_quantum")),
        "quantum_framework": state.get("quantum_framework"),
        "quantum_algorithm": state.get("quantum_algorithm"),
        "quantum_qubit_count": state.get("quantum_qubit_count"),
        "quantum_backend": state.get("quantum_backend"),
        "dataset_source": state.get("dataset_source"),
        "dataset_path": state.get("dataset_path"),
        "kaggle_dataset_id": state.get("kaggle_dataset_id"),
        "hardware_target": state.get("hardware_target"),
        "max_epochs": state.get("max_epochs"),
        "batch_size": state.get("batch_size"),
        "random_seed": state.get("random_seed"),
        "required_packages": list(state.get("required_packages") or []),
        "allowed_import_roots": sorted(_build_allowed_import_roots(state)),
        "data_report": state.get("data_report") or {},
        "clarifications": state.get("clarifications") or {},
        "research_plan": state.get("research_plan") or {},
    }


async def _invoke_codegen_llm(
    *,
    state: ResearchState,
    context: dict[str, Any],
    required_relative_files: list[str],
    repair_violations: list[str] | None = None,
    existing_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    repair_violations = list(repair_violations or [])
    system_prompt = (
        "SYSTEM ROLE: codegen_file_manifest.\n"
        "Generate project source files dynamically from user state.\n"
        "Return one JSON object only with keys:\n"
        "- problem_type (string)\n"
        "- code_level (string)\n"
        "- algorithm_class (string)\n"
        "- files (array of {\"path\": \"relative/path.py\", \"content\": \"...\"})\n"
        "Rules:\n"
        "- problem_type, code_level, algorithm_class MUST exactly match provided state contract.\n"
        "- Use only allowed import roots from state context.\n"
        "- No placeholders, no TODO, no pseudo-code.\n"
        "- Do not use unsafe patterns (eval, exec, os.system, shell=True).\n"
        "- Keep paths strictly under project root and use relative paths.\n"
        "- config.py must include PROBLEM_TYPE and CODE_LEVEL constants.\n"
        "- main.py must be runnable entrypoint for the generated project.\n"
        "- Preprocessing must adapt to state.data_report and detected problem_type; avoid fixed hard-coded preprocessing steps.\n"
    )
    if repair_violations:
        system_prompt += "This is a strict-state repair turn. Fix every listed violation.\n"

    prompt_parts = [
        f"Required file set:\n{json.dumps(required_relative_files, indent=2)}",
        f"State context:\n{json.dumps(context, indent=2, default=str)}",
    ]
    if existing_files:
        prompt_parts.append(
            f"Existing generated files (absolute_path -> content):\n{json.dumps(existing_files, indent=2, default=str)}"
        )
    if repair_violations:
        prompt_parts.append(f"Strict violations to fix exactly:\n{json.dumps(repair_violations, indent=2)}")
    prompt_parts.append("Return JSON only.")
    user_prompt = "\n\n".join(prompt_parts)

    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="code_generator",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = parse_json_object(raw)
    if not parsed:
        logger.warning("agent.codegen.dynamic_parse_failed", experiment_id=state["experiment_id"])
    return parsed


async def _generate_dynamic_files(
    state: ResearchState,
    project: Path,
    problem_type: str,
    code_level: str,
    algorithm_class: str,
) -> dict[str, str]:
    context = _build_codegen_context(state, problem_type, code_level, algorithm_class)
    required_relative_files = list(_REQUIRED_RELATIVE_FILES)
    repair_attempts = max(0, int(settings.STRICT_STATE_ONLY_REPAIR_ATTEMPTS))

    payload = await _invoke_codegen_llm(
        state=state,
        context=context,
        required_relative_files=required_relative_files,
    )
    manifest = _normalize_manifest(project, _extract_file_items(payload))
    _ensure_config_contract(
        manifest=manifest,
        project=project,
        problem_type=problem_type,
        code_level=code_level,
        algorithm_class=algorithm_class,
        target_metric=str(state.get("target_metric") or "accuracy"),
        hardware_target=str(state.get("hardware_target") or "cpu"),
    )

    violations = _collect_strict_violations(
        state=state,
        payload=payload,
        manifest=manifest,
        project=project,
        expected_problem_type=problem_type,
        expected_code_level=code_level,
        expected_algorithm_class=algorithm_class,
    )
    state.setdefault("research_plan", {})["codegen_strict_violations"] = list(violations)
    if violations:
        logger.warning("agent.codegen.dynamic_validation_failed", experiment_id=state["experiment_id"], violations=violations)

    attempts_used = 0
    while violations and attempts_used < repair_attempts:
        attempts_used += 1
        payload = await _invoke_codegen_llm(
            state=state,
            context=context,
            required_relative_files=required_relative_files,
            repair_violations=violations,
            existing_files=manifest,
        )
        repaired_manifest = _normalize_manifest(project, _extract_file_items(payload))
        if repaired_manifest:
            manifest.update(repaired_manifest)
        _ensure_config_contract(
            manifest=manifest,
            project=project,
            problem_type=problem_type,
            code_level=code_level,
            algorithm_class=algorithm_class,
            target_metric=str(state.get("target_metric") or "accuracy"),
            hardware_target=str(state.get("hardware_target") or "cpu"),
        )
        violations = _collect_strict_violations(
            state=state,
            payload=payload,
            manifest=manifest,
            project=project,
            expected_problem_type=problem_type,
            expected_code_level=code_level,
            expected_algorithm_class=algorithm_class,
        )
        state.setdefault("research_plan", {})["codegen_strict_violations"] = list(violations)
        if violations:
            logger.warning(
                "agent.codegen.dynamic_validation_failed",
                experiment_id=state["experiment_id"],
                attempt=attempts_used,
                violations=violations,
            )

    if violations:
        raise RuntimeError(f"strict_state_only violation after {attempts_used} repair attempts: {violations}")
    return manifest


async def code_gen_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "code_generator"
    logger.info("agent.codegen.start", experiment_id=state["experiment_id"], requires_quantum=state["requires_quantum"])
    local_mode = is_vscode_execution_mode(state)
    project = Path(state["project_path"])

    problem_type = _resolve_problem_type(state)
    code_level = _resolve_code_level(state)
    algorithm_class = _resolve_algorithm_class(state)
    state.setdefault("research_plan", {})["problem_type"] = problem_type
    state["research_plan"]["code_level"] = code_level
    state["research_plan"]["algorithm_class"] = algorithm_class

    required_dirs = [
        "src",
        "data/raw",
        "data/processed",
        "outputs/plots",
        "outputs/model_checkpoint",
        "logs",
        "docs",
    ]
    for rel in required_dirs:
        (project / rel).mkdir(parents=True, exist_ok=True)

    try:
        files = await _generate_dynamic_files(
            state=state,
            project=project,
            problem_type=problem_type,
            code_level=code_level,
            algorithm_class=algorithm_class,
        )
    except Exception as exc:
        logger.exception("agent.codegen.dynamic_generation_failed", experiment_id=state["experiment_id"])
        raise RuntimeError(f"Dynamic code generation failed: {exc}") from exc

    if _should_inject_failure("codegen_syntax"):
        main_path = str((project / "main.py").resolve())
        files[main_path] = f"{files.get(main_path, '')}\nthis is invalid python\n"

    if local_mode:
        plan = state.setdefault("local_file_plan", [])
        planned_files: list[dict[str, str]] = []
        for path, content in files.items():
            item = {"path": path, "content": content, "phase": "code_generator"}
            plan.append(item)
            planned_files.append(item)
            if path not in state["created_files"]:
                state["created_files"].append(path)
            logger.info("agent.codegen.file_planned", experiment_id=state["experiment_id"], file_path=path, size=len(content))
        next_phase = "quantum_gate" if state["requires_quantum"] else "job_scheduler"
        queued = queue_local_file_action(
            state=state,
            phase="code_generator",
            file_operations=planned_files,
            next_phase=next_phase,
            reason=f"Create dynamically generated source files locally ({problem_type}/{code_level}) before scheduling execution",
            cwd=state["project_path"],
        )
        if queued:
            logger.info(
                "agent.codegen.pending_local_action",
                experiment_id=state["experiment_id"],
                file_count=len(planned_files),
                next_phase=next_phase,
            )
            return state
    else:
        for path, content in files.items():
            write_text_file(state["project_path"], path, content)
            if path not in state["created_files"]:
                state["created_files"].append(path)
            logger.info("agent.codegen.file_written", experiment_id=state["experiment_id"], file_path=path, size=len(content))

    logger.info(
        "agent.codegen.end",
        experiment_id=state["experiment_id"],
        created_files=len(files),
        problem_type=problem_type,
        code_level=code_level,
    )
    return state
