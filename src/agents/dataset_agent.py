from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from src.config.settings import settings
from src.core.file_manager import write_text_file
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

def _synthetic_dataset(state: ResearchState, path: Path) -> dict:
    rng = random.Random(state["random_seed"])
    rows = []
    for _ in range(200):
        f1 = rng.random()
        f2 = rng.random()
        f3 = rng.random()
        target = int((f1 + f2 + f3) > 1.5)
        rows.append((f1, f2, f3, target))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_1", "feature_2", "feature_3", "target"])
        writer.writerows(rows)

    return {
        "filename": path.name,
        "shape": [len(rows), 4],
        "columns": ["feature_1", "feature_2", "feature_3", "target"],
        "dtypes": {"feature_1": "float", "feature_2": "float", "feature_3": "float", "target": "int"},
        "null_counts": {"feature_1": 0, "feature_2": 0, "feature_3": 0, "target": 0},
        "class_distribution": {"0": sum(1 for r in rows if r[3] == 0), "1": sum(1 for r in rows if r[3] == 1)},
        "sample_rows": [
            {"feature_1": rows[0][0], "feature_2": rows[0][1], "feature_3": rows[0][2], "target": rows[0][3]},
            {"feature_1": rows[1][0], "feature_2": rows[1][1], "feature_3": rows[1][2], "target": rows[1][3]},
        ],
    }


def _validation_script(project_path: str) -> str:
    return f"""from __future__ import annotations
import csv
import json
from pathlib import Path

raw_dir = Path(r\"{project_path}\") / \"data\" / \"raw\"
csv_files = list(raw_dir.glob(\"*.csv\"))
if not csv_files:
    raise SystemExit(\"DATA_ERROR: No CSV files found\")
path = csv_files[0]
rows = []
with path.open(\"r\", encoding=\"utf-8\") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        rows.append(row)
columns = list(rows[0].keys()) if rows else []
report = {{
    \"filename\": path.name,
    \"shape\": [len(rows), len(columns)],
    \"columns\": columns,
    \"sample_rows\": rows[:5]
}}
out_path = Path(r\"{project_path}\") / \"data\" / \"data_report.json\"
out_path.write_text(json.dumps(report, indent=2), encoding=\"utf-8\")
print(\"DATA_REPORT:\", json.dumps(report))
print(\"DATA_VALID: true\")
"""


async def dataset_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "dataset_manager"
    logger.info("agent.dataset.start", experiment_id=state["experiment_id"], source=state["dataset_source"])
    project = Path(state["project_path"])
    raw_dir = project / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = raw_dir / "dataset.csv"

    report = _synthetic_dataset(state, dataset_csv)
    if _should_inject_failure("dataset_corruption"):
        dataset_csv.write_text("bad_col\ncorrupted_row\n", encoding="utf-8")
        report["injected_failure"] = "dataset_corruption"
    state["dataset_path"] = str(raw_dir)
    state["data_report"] = report

    # Keep generated scripts as artifacts for traceability.
    validate_path = project / "data" / "validate_data.py"
    write_text_file(state["project_path"], str(validate_path), _validation_script(state["project_path"]))
    state["created_files"].append(str(validate_path))

    (project / "data" / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "agent.dataset.end",
        experiment_id=state["experiment_id"],
        dataset_path=str(raw_dir),
        shape=report.get("shape"),
    )
    return state
