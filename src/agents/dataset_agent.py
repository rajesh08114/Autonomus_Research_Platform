from __future__ import annotations

import csv
import json
import random
import re
from io import StringIO
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode, local_python_command
from src.core.file_manager import write_text_file
from src.core.local_actions import queue_local_file_action
from src.core.logger import get_logger
from src.llm.dynamic_parser import parse_json_object
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ResearchState

logger = get_logger(__name__)

_ALLOWED_CHECK_TYPES = {"min_rows", "required_columns", "target_binary", "target_numeric"}
_SAFE_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _should_inject_failure(point: str) -> bool:
    if not settings.FAILURE_INJECTION_ENABLED:
        return False
    configured = {p.strip().lower() for p in settings.FAILURE_INJECTION_POINTS.split(",") if p.strip()}
    if configured and point.lower() not in configured:
        return False
    return random.random() < float(max(0.0, min(1.0, settings.FAILURE_INJECTION_RATE)))


def _rows_to_csv(rows: list[tuple[float, float, float, int]]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["feature_1", "feature_2", "feature_3", "target"])
    writer.writerows(rows)
    return buffer.getvalue()


def _resolve_problem_type(state: ResearchState) -> str:
    from_plan = str((state.get("research_plan") or {}).get("problem_type", "")).strip().lower()
    from_clar = str((state.get("clarifications") or {}).get("problem_type", "")).strip().lower()
    metric = str(state.get("target_metric", "")).strip().lower()
    if from_plan in {"classification", "regression", "clustering", "reinforcement", "forecasting", "generation"}:
        return from_plan
    if from_clar in {"classification", "regression", "clustering", "reinforcement", "forecasting", "generation"}:
        return from_clar
    if metric in {"rmse", "mae", "mse", "r2"}:
        return "regression"
    return "classification"


def _synthetic_rows(state: ResearchState) -> list[tuple[float, float, float, int | float]]:
    rng = random.Random(state["random_seed"])
    problem_type = _resolve_problem_type(state)
    rows: list[tuple[float, float, float, int | float]] = []
    for _ in range(200):
        f1 = rng.random()
        f2 = rng.random()
        f3 = rng.random()
        if problem_type == "regression":
            target = round((f1 * 4.5) + (f2 * 3.2) + (f3 * 2.3) + (rng.random() * 0.1), 6)
        elif problem_type == "clustering":
            target = int((f1 > 0.5 and f2 > 0.5) or f3 > 0.7)
        else:
            target = int((f1 + f2 + f3) > 1.5)
        rows.append((f1, f2, f3, target))
    return rows


def _sklearn_rows(state: ResearchState) -> list[tuple[float, float, float, int | float]]:
    try:
        problem_type = _resolve_problem_type(state)
        if problem_type == "regression":
            from sklearn.datasets import load_diabetes

            ds = load_diabetes(as_frame=True)
        else:
            from sklearn.datasets import load_wine

            ds = load_wine(as_frame=True)
        frame = ds.frame
        if frame is None or frame.empty:
            raise RuntimeError("sklearn dataset is empty")
        feature_names = list(ds.feature_names[:3]) if ds.feature_names else list(frame.columns[:3])
        rows: list[tuple[float, float, float, int | float]] = []
        for _, row in frame.iterrows():
            f1 = float(row.get(feature_names[0], 0.0))
            f2 = float(row.get(feature_names[1], 0.0))
            f3 = float(row.get(feature_names[2], 0.0))
            if problem_type == "regression":
                target = float(row.get("target", 0.0))
            else:
                target = 1 if int(row.get("target", 0)) > 0 else 0
            rows.append((f1, f2, f3, target))
        if not rows:
            raise RuntimeError("sklearn dataset has no rows")
        return rows
    except Exception as exc:
        raise RuntimeError(f"sklearn dataset load failed: {exc}") from exc


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_target(value: Any) -> int:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "positive", "pos"}:
        return 1
    try:
        return 1 if float(text) > 0 else 0
    except Exception:
        return 0


def _rows_from_csv(path: Path, problem_type: str) -> list[tuple[float, float, float, int | float]]:
    rows: list[tuple[float, float, float, int | float]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            source_rows = [dict(item) for item in reader]
    except Exception:
        return rows
    if not source_rows:
        return rows

    candidate_columns = [str(name).strip() for name in source_rows[0].keys() if str(name).strip()]
    target_candidates = {"target", "label", "class", "y", "outcome"}
    target_col = next((name for name in candidate_columns if name.lower() in target_candidates), "")
    if not target_col and candidate_columns:
        target_col = candidate_columns[-1]

    feature_cols: list[str] = []
    for name in candidate_columns:
        if name == target_col:
            continue
        feature_cols.append(name)
        if len(feature_cols) >= 3:
            break
    while len(feature_cols) < 3:
        feature_cols.append("")

    for row in source_rows:
        features: list[float] = []
        for col in feature_cols:
            number = _to_float(row.get(col))
            features.append(float(number if number is not None else 0.0))
        if problem_type == "regression":
            label = float(_to_float(row.get(target_col)) or 0.0)
        else:
            label = _to_target(row.get(target_col))
        rows.append((features[0], features[1], features[2], label))
    return rows


def _find_external_csv(raw_dir: Path) -> Path | None:
    candidates = [path for path in sorted(raw_dir.glob("*.csv")) if path.name.lower() != "dataset.csv"]
    return candidates[0] if candidates else None


def _select_rows_for_source(state: ResearchState, raw_dir: Path) -> tuple[list[tuple[float, float, float, int | float]], dict[str, Any]]:
    source = str(state.get("dataset_source", "synthetic")).strip().lower() or "synthetic"
    problem_type = _resolve_problem_type(state)
    metadata: dict[str, Any] = {"requested_source": source, "resolved_source": source, "problem_type": problem_type}
    if source == "synthetic":
        return _synthetic_rows(state), metadata
    if source == "sklearn":
        return _sklearn_rows(state), metadata
    if source in {"upload", "kaggle"}:
        external_path = _find_external_csv(raw_dir)
        if external_path:
            loaded = _rows_from_csv(external_path, problem_type=problem_type)
            if loaded:
                metadata["resolved_source"] = source
                metadata["source_file"] = str(external_path)
                return loaded, metadata
            raise RuntimeError(f"{source} CSV parse failed: {external_path.name}")
        raise RuntimeError(f"{source} dataset source requires a CSV file in data/raw")
    raise RuntimeError(f"Unsupported dataset source: {source}")


def _default_dataset_plan(problem_type: str) -> dict[str, Any]:
    target_check = "target_numeric" if problem_type == "regression" else "target_binary"
    return {
        "schema_mapping": {
            "feature_1": "feature_1",
            "feature_2": "feature_2",
            "feature_3": "feature_3",
            "target": "target",
        },
        "validation_checks": [
            {"type": "required_columns", "value": ["feature_1", "feature_2", "feature_3", "target"]},
            {"type": "min_rows", "value": 10},
            {"type": target_check, "value": True},
        ],
        "report_narrative": {
            "summary": "Dataset was normalized to feature_1/feature_2/feature_3/target contract.",
            "quality_notes": ["Schema validated against required columns and minimum row count."],
        },
    }


async def _invoke_dataset_plan_llm(state: ResearchState, source_meta: dict[str, Any]) -> dict[str, Any]:
    system_prompt = (
        "SYSTEM ROLE: dataset_dynamic_plan.\n"
        "Return JSON only with keys:\n"
        "- schema_mapping: object with keys feature_1, feature_2, feature_3, target and string values.\n"
        "- validation_checks: array of objects with keys type and value.\n"
        "- report_narrative: object with keys summary (string) and quality_notes (string array).\n"
        "Allowed validation check types: min_rows, required_columns, target_binary, target_numeric.\n"
        "Do not return executable code."
    )
    user_prompt = json.dumps(
        {
            "dataset_source": state.get("dataset_source"),
            "framework": state.get("framework"),
            "problem_type": _resolve_problem_type(state),
            "target_metric": state.get("target_metric"),
            "required_packages": state.get("required_packages"),
            "research_plan": state.get("research_plan", {}),
            "source_meta": source_meta,
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="dataset_manager",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = parse_json_object(raw)
    if not parsed:
        logger.warning("agent.dataset.dynamic_parse_failed", experiment_id=state["experiment_id"])
    return parsed


def _sanitize_dataset_plan(plan: dict[str, Any], problem_type: str) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    mapping_raw = plan.get("schema_mapping")
    mapping: dict[str, str] = {}
    if not isinstance(mapping_raw, dict):
        violations.append("schema_mapping must be an object")
    else:
        for key in ("feature_1", "feature_2", "feature_3", "target"):
            value = str(mapping_raw.get(key, "")).strip()
            if not value:
                violations.append(f"schema_mapping missing value for {key}")
                continue
            if not _SAFE_COLUMN_RE.match(value):
                violations.append(f"schema_mapping has unsafe column name for {key}: {value}")
                continue
            mapping[key] = value

    checks: list[dict[str, Any]] = []
    checks_raw = plan.get("validation_checks")
    if not isinstance(checks_raw, list):
        violations.append("validation_checks must be an array")
    else:
        for idx, item in enumerate(checks_raw):
            if not isinstance(item, dict):
                violations.append(f"validation_checks[{idx}] must be an object")
                continue
            check_type = str(item.get("type", "")).strip().lower()
            if check_type not in _ALLOWED_CHECK_TYPES:
                violations.append(f"validation_checks[{idx}] unsupported type: {check_type}")
                continue
            value = item.get("value")
            if check_type == "min_rows":
                try:
                    value = int(value)
                except Exception:
                    violations.append("min_rows value must be integer")
                    continue
                value = max(1, min(value, 1_000_000))
            elif check_type == "required_columns":
                if not isinstance(value, list):
                    violations.append("required_columns value must be array")
                    continue
                clean_cols: list[str] = []
                for col in value:
                    col_name = str(col).strip()
                    if col_name and _SAFE_COLUMN_RE.match(col_name):
                        clean_cols.append(col_name)
                if not clean_cols:
                    violations.append("required_columns must include at least one safe column")
                    continue
                value = clean_cols[:20]
            elif check_type in {"target_binary", "target_numeric"}:
                value = bool(value)
            checks.append({"type": check_type, "value": value})

    narrative_raw = plan.get("report_narrative")
    narrative: dict[str, Any] = {"summary": "", "quality_notes": []}
    if not isinstance(narrative_raw, dict):
        violations.append("report_narrative must be an object")
    else:
        summary = str(narrative_raw.get("summary", "")).strip()
        notes_raw = narrative_raw.get("quality_notes", [])
        notes = [str(item).strip() for item in notes_raw] if isinstance(notes_raw, list) else []
        notes = [item for item in notes if item][:8]
        narrative = {"summary": summary[:500], "quality_notes": [item[:200] for item in notes]}

    target_check = "target_numeric" if problem_type == "regression" else "target_binary"
    if not any(str(item.get("type")) == target_check for item in checks):
        checks.append({"type": target_check, "value": True})
    if not any(str(item.get("type")) == "required_columns" for item in checks):
        checks.append({"type": "required_columns", "value": ["feature_1", "feature_2", "feature_3", "target"]})
    if not any(str(item.get("type")) == "min_rows" for item in checks):
        checks.append({"type": "min_rows", "value": 10})

    sanitized = {"schema_mapping": mapping, "validation_checks": checks, "report_narrative": narrative}
    return sanitized, violations


def _build_validation_script(project_path: str, plan: dict[str, Any]) -> str:
    checks_json = json.dumps(plan.get("validation_checks", []), indent=2)
    return f"""from __future__ import annotations
import csv
import json
from pathlib import Path

raw_dir = Path(r\"{project_path}\") / "data" / "raw"
path = raw_dir / "dataset.csv"
if not path.exists():
    raise SystemExit("DATA_ERROR: dataset.csv not found")

with path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = [dict(item) for item in reader]

columns = list(rows[0].keys()) if rows else []
checks = json.loads(r'''{checks_json}''')

for check in checks:
    ctype = str(check.get("type", ""))
    cvalue = check.get("value")
    if ctype == "min_rows":
        if len(rows) < int(cvalue):
            raise SystemExit(f"DATA_ERROR: expected at least {{int(cvalue)}} rows, got {{len(rows)}}")
    elif ctype == "required_columns":
        required = [str(item) for item in (cvalue or [])]
        missing = [name for name in required if name not in columns]
        if missing:
            raise SystemExit(f"DATA_ERROR: missing columns {{missing}}")
    elif ctype == "target_binary":
        if rows and bool(cvalue):
            bad = []
            for idx, row in enumerate(rows[:1000]):
                value = str(row.get("target", "")).strip()
                if value not in {{"0", "1", "0.0", "1.0", "true", "false", "True", "False"}}:
                    bad.append(idx)
                    if len(bad) >= 5:
                        break
            if bad:
                raise SystemExit(f"DATA_ERROR: non-binary target values at rows {{bad}}")
    elif ctype == "target_numeric":
        if rows and bool(cvalue):
            bad = []
            for idx, row in enumerate(rows[:1000]):
                value = str(row.get("target", "")).strip()
                try:
                    float(value)
                except Exception:
                    bad.append(idx)
                    if len(bad) >= 5:
                        break
            if bad:
                raise SystemExit(f"DATA_ERROR: non-numeric target values at rows {{bad}}")

report = {{
    "filename": path.name,
    "shape": [len(rows), len(columns)],
    "columns": columns,
    "sample_rows": rows[:5]
}}
out_path = Path(r\"{project_path}\") / "data" / "data_report.json"
out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("DATA_REPORT:", json.dumps(report))
print("DATA_VALID: true")
"""


def _dataset_report(
    path: Path,
    rows: list[tuple[float, float, float, int | float]],
    source_meta: dict[str, Any],
    dataset_plan: dict[str, Any],
    used_dynamic_plan: bool,
) -> dict[str, Any]:
    problem_type = str(source_meta.get("problem_type", "classification"))
    target_dtype = "float" if problem_type == "regression" else "int"
    sample: list[dict[str, Any]] = []
    if rows:
        sample.append({"feature_1": rows[0][0], "feature_2": rows[0][1], "feature_3": rows[0][2], "target": rows[0][3]})
    if len(rows) > 1:
        sample.append({"feature_1": rows[1][0], "feature_2": rows[1][1], "feature_3": rows[1][2], "target": rows[1][3]})
    report: dict[str, Any] = {
        "filename": path.name,
        "shape": [len(rows), 4],
        "columns": ["feature_1", "feature_2", "feature_3", "target"],
        "dtypes": {"feature_1": "float", "feature_2": "float", "feature_3": "float", "target": target_dtype},
        "null_counts": {"feature_1": 0, "feature_2": 0, "feature_3": 0, "target": 0},
        "class_distribution": {"0": sum(1 for row in rows if float(row[3]) <= 0.5), "1": sum(1 for row in rows if float(row[3]) > 0.5)},
        "sample_rows": sample,
        "source": source_meta,
        "dynamic_plan": {
            "enabled": bool(settings.DATASET_DYNAMIC_ENABLED),
            "used_dynamic_plan": used_dynamic_plan,
            "schema_mapping": dataset_plan.get("schema_mapping", {}),
            "validation_checks": dataset_plan.get("validation_checks", []),
            "report_narrative": dataset_plan.get("report_narrative", {}),
        },
    }
    return report


def _kaggle_download_script(project_path: str, dataset_id: str) -> str:
    return f"""from __future__ import annotations
import subprocess
from pathlib import Path

project = Path(r\"{project_path}\")
raw_dir = project / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
cmd = ["kaggle", "datasets", "download", "-d", "{dataset_id}", "-p", str(raw_dir), "--unzip"]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", result.returncode)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
if result.returncode != 0:
    raise SystemExit(f"KAGGLE_DOWNLOAD_FAILED: {{result.stderr[:500]}}")
"""


async def dataset_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "dataset_manager"
    requested_source = str(state.get("dataset_source", "synthetic")).strip().lower() or "synthetic"
    logger.info("agent.dataset.start", experiment_id=state["experiment_id"], source=requested_source)
    local_mode = is_vscode_execution_mode(state)
    project = Path(state["project_path"])
    raw_dir = project / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = raw_dir / "dataset.csv"

    rows, source_meta = _select_rows_for_source(state, raw_dir)
    problem_type = _resolve_problem_type(state)
    dataset_plan = _default_dataset_plan(problem_type)
    used_dynamic_plan = False
    fallback_static = False

    if settings.DATASET_DYNAMIC_ENABLED:
        plan_payload = await _invoke_dataset_plan_llm(state, source_meta)
        if not plan_payload:
            if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                logger.warning("agent.dataset.dynamic_fallback_static", experiment_id=state["experiment_id"], reason="parse_failed")
                fallback_static = True
            else:
                raise RuntimeError("Dataset dynamic planning failed: empty/invalid LLM JSON response")
        else:
            parsed_plan, violations = _sanitize_dataset_plan(plan_payload, problem_type)
            if violations:
                logger.warning("agent.dataset.dynamic_validation_failed", experiment_id=state["experiment_id"], violations=violations)
                if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                    logger.warning(
                        "agent.dataset.dynamic_fallback_static",
                        experiment_id=state["experiment_id"],
                        reason="validation_failed",
                    )
                    fallback_static = True
                else:
                    raise RuntimeError(f"Dataset dynamic plan validation failed: {violations}")
            else:
                dataset_plan = parsed_plan
                used_dynamic_plan = True
    state.setdefault("research_plan", {})["dataset_dynamic_plan_summary"] = {
        "enabled": bool(settings.DATASET_DYNAMIC_ENABLED),
        "used_dynamic_plan": used_dynamic_plan,
        "fallback_static": fallback_static,
        "validation_checks": dataset_plan.get("validation_checks", []),
        "schema_mapping": dataset_plan.get("schema_mapping", {}),
    }

    report = _dataset_report(dataset_csv, rows, source_meta, dataset_plan, used_dynamic_plan)
    if "shape" not in report or "columns" not in report:
        raise RuntimeError("Dataset report contract violation: missing shape/columns")
    if "source" not in report:
        report["source"] = source_meta
    csv_text = _rows_to_csv(rows)
    if _should_inject_failure("dataset_corruption"):
        csv_text = "bad_col\ncorrupted_row\n"
        report["injected_failure"] = "dataset_corruption"

    state["dataset_path"] = str(raw_dir)
    state["data_report"] = report

    validate_path = project / "data" / "validate_data.py"
    planned_files = [
        {"path": str(dataset_csv), "content": csv_text, "phase": "dataset_manager"},
        {"path": str(validate_path), "content": _build_validation_script(state["project_path"], dataset_plan), "phase": "dataset_manager"},
        {
            "path": str(project / "data" / "data_report.json"),
            "content": json.dumps(report, indent=2),
            "phase": "dataset_manager",
        },
    ]
    if requested_source == "kaggle" and state.get("kaggle_dataset_id"):
        planned_files.append(
            {
                "path": str(project / "data" / "download_kaggle.py"),
                "content": _kaggle_download_script(state["project_path"], str(state["kaggle_dataset_id"])),
                "phase": "dataset_manager",
            }
        )

    if local_mode:
        plan = state.setdefault("local_file_plan", [])
        plan.extend(planned_files)
        for item in planned_files:
            if item["path"] not in state["created_files"]:
                state["created_files"].append(item["path"])
        queued = queue_local_file_action(
            state=state,
            phase="dataset_manager",
            file_operations=planned_files,
            next_phase="code_generator",
            reason=f"Create dataset artifacts locally for source={requested_source}",
            commands=[local_python_command(), str(validate_path)],
            cwd=state["project_path"],
            timeout_seconds=int(settings.SUBPROCESS_TIMEOUT),
        )
        if queued:
            logger.info(
                "agent.dataset.pending_local_action",
                experiment_id=state["experiment_id"],
                file_count=len(planned_files),
                next_phase="code_generator",
                resolved_source=source_meta.get("resolved_source"),
            )
            return state
    else:
        for item in planned_files:
            write_text_file(state["project_path"], item["path"], item["content"])
            if item["path"] not in state["created_files"]:
                state["created_files"].append(item["path"])

    logger.info(
        "agent.dataset.end",
        experiment_id=state["experiment_id"],
        dataset_path=str(raw_dir),
        shape=report.get("shape"),
        resolved_source=(report.get("source") or {}).get("resolved_source"),
    )
    return state
