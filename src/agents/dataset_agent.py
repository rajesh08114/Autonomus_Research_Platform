from __future__ import annotations

import csv
import json
import random
from io import StringIO
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode, local_python_command
from src.core.file_manager import write_text_file
from src.core.local_actions import queue_local_file_action
from src.core.logger import get_logger
from src.state.research_state import ResearchState

logger = get_logger(__name__)


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


def _sklearn_rows(state: ResearchState) -> tuple[list[tuple[float, float, float, int | float]], str | None]:
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
            return _synthetic_rows(state), "sklearn_dataset_empty_fallback_to_synthetic"
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
            return _synthetic_rows(state), "sklearn_no_rows_fallback_to_synthetic"
        return rows, None
    except Exception:
        return _synthetic_rows(state), "sklearn_import_fallback_to_synthetic"


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
        rows, note = _sklearn_rows(state)
        if note:
            metadata["source_note"] = note
            metadata["resolved_source"] = "synthetic"
        return rows, metadata
    if source in {"upload", "kaggle"}:
        external_path = _find_external_csv(raw_dir)
        if external_path:
            loaded = _rows_from_csv(external_path, problem_type=problem_type)
            if loaded:
                metadata["resolved_source"] = source
                metadata["source_file"] = str(external_path)
                return loaded, metadata
            metadata["source_note"] = "external_csv_parse_failed_fallback_to_synthetic"
        else:
            metadata["source_note"] = f"{source}_dataset_not_found_fallback_to_synthetic"
        metadata["resolved_source"] = "synthetic"
        return _synthetic_rows(state), metadata
    metadata["source_note"] = "unknown_dataset_source_fallback_to_synthetic"
    metadata["resolved_source"] = "synthetic"
    return _synthetic_rows(state), metadata


def _dataset_report(path: Path, rows: list[tuple[float, float, float, int | float]], source_meta: dict[str, Any]) -> dict[str, Any]:
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
    }
    return report


def _kaggle_download_script(project_path: str, dataset_id: str) -> str:
    return f"""from __future__ import annotations
import os
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


def _validation_script(project_path: str) -> str:
    return f"""from __future__ import annotations
import csv
import json
from pathlib import Path

raw_dir = Path(r\"{project_path}\") / "data" / "raw"
csv_files = list(raw_dir.glob("*.csv"))
if not csv_files:
    raise SystemExit("DATA_ERROR: No CSV files found")
path = csv_files[0]
rows = []
with path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        rows.append(row)
columns = list(rows[0].keys()) if rows else []
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
    report = _dataset_report(dataset_csv, rows, source_meta)
    csv_text = _rows_to_csv(rows)
    if _should_inject_failure("dataset_corruption"):
        csv_text = "bad_col\ncorrupted_row\n"
        report["injected_failure"] = "dataset_corruption"

    state["dataset_path"] = str(raw_dir)
    state["data_report"] = report

    validate_path = project / "data" / "validate_data.py"
    planned_files = [
        {"path": str(dataset_csv), "content": csv_text, "phase": "dataset_manager"},
        {"path": str(validate_path), "content": _validation_script(state["project_path"]), "phase": "dataset_manager"},
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
