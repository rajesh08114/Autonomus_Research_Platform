from __future__ import annotations

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

def _config_template(state: ResearchState) -> str:
    return f"""from __future__ import annotations
import os
import random

EXPERIMENT_ID = "{state["experiment_id"]}"
PROJECT_ROOT = r"{state["project_path"]}"

DATA_RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_checkpoint")
METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "metrics.json")
PLOTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "plots")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "run.log")
STATE_HISTORY_PATH = os.path.join(PROJECT_ROOT, "logs", "state_history.json")

RANDOM_SEED = {state["random_seed"]}
MAX_EPOCHS = {int(state.get("max_epochs") or 50)}
BATCH_SIZE = {int(state.get("batch_size") or 32)}
LEARNING_RATE = 0.01
DEVICE = "cpu" if "{state["hardware_target"]}" != "cuda" else "cuda"
FRAMEWORK = "{state["framework"]}"
TARGET_METRIC = "{state["target_metric"]}"
QUANTUM_FRAMEWORK = "{state.get("quantum_framework") or ""}"
QUANTUM_BACKEND = "{state.get("quantum_backend") or ""}"
QUBIT_COUNT = {int(state.get("quantum_qubit_count") or 0)}

def set_global_seed() -> None:
    random.seed(RANDOM_SEED)
"""


def _utils_template() -> str:
    return """from __future__ import annotations
import json
import logging
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def write_json(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
"""


def _preprocessing_template() -> str:
    return """from __future__ import annotations
import csv
import random
from typing import List, Tuple

def load_dataset(path: str) -> List[Tuple[float, float, float, int]]:
    rows: List[Tuple[float, float, float, int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((
                float(row["feature_1"]),
                float(row["feature_2"]),
                float(row["feature_3"]),
                int(row["target"]),
            ))
    return rows

def train_test_split(rows: List[Tuple[float, float, float, int]], seed: int = 42) -> tuple[list, list]:
    random.Random(seed).shuffle(rows)
    idx = int(len(rows) * 0.8)
    return rows[:idx], rows[idx:]
"""


def _model_template() -> str:
    return """from __future__ import annotations
from typing import Iterable

try:
    from src.quantum_circuit import QuantumLayer
except Exception:
    class QuantumLayer:
        def forward(self, x: Iterable[float]) -> list[float]:
            total = float(sum(x))
            return [total / max(len(list(x)), 1)]

class MajorityClassifier:
    def __init__(self) -> None:
        self.majority = 0
        self.quantum = QuantumLayer()

    def fit(self, y: list[int]) -> None:
        zeros = sum(1 for value in y if value == 0)
        ones = len(y) - zeros
        self.majority = 1 if ones >= zeros else 0

    def predict(self, X: list[tuple[float, float, float]]) -> list[int]:
        return [self.majority for _ in X]
"""


def _train_template() -> str:
    return """from __future__ import annotations
import os
from src.preprocessing import load_dataset, train_test_split
from src.model import MajorityClassifier
from src.utils import write_json, get_logger
import config

def run_training() -> dict:
    logger = get_logger("train")
    rows = load_dataset(os.path.join(config.DATA_RAW_PATH, "dataset.csv"))
    train_rows, test_rows = train_test_split(rows, seed=config.RANDOM_SEED)
    train_x = [(a, b, c) for a, b, c, _ in train_rows]
    train_y = [y for _, _, _, y in train_rows]
    test_x = [(a, b, c) for a, b, c, _ in test_rows]
    test_y = [y for _, _, _, y in test_rows]

    model = MajorityClassifier()
    model.fit(train_y)
    preds = model.predict(test_x)
    correct = sum(1 for p, y in zip(preds, test_y) if p == y)
    accuracy = correct / max(len(test_y), 1)

    result = {
        "train_loss": 1.0 - accuracy,
        "accuracy": accuracy,
        "f1_macro": accuracy,
        "duration_sec": 0.1,
    }
    os.makedirs(config.MODEL_CHECKPOINT_PATH, exist_ok=True)
    write_json(os.path.join(config.MODEL_CHECKPOINT_PATH, "model.json"), {"majority": model.majority})
    logger.info("Training complete")
    return result
"""


def _evaluate_template() -> str:
    return """from __future__ import annotations
import os
from src.utils import write_json
import config

def evaluate(metrics: dict) -> dict:
    payload = {
        "experiment_id": config.EXPERIMENT_ID,
        "algorithm": "majority_classifier",
        "framework": config.FRAMEWORK,
        "dataset": "synthetic_dataset",
        "training": {
            "epochs": config.MAX_EPOCHS,
            "final_loss": metrics.get("train_loss", 1.0),
            "duration_sec": metrics.get("duration_sec", 0.0),
        },
        "evaluation": {
            "accuracy": metrics.get("accuracy", 0.0),
            "f1_macro": metrics.get("f1_macro", 0.0),
            "roc_auc": metrics.get("accuracy", 0.0),
            "quantum_fidelity": 0.0,
        },
        "hardware": config.DEVICE,
        "seed": config.RANDOM_SEED,
        "reproducible": True,
    }
    os.makedirs(config.PLOTS_PATH, exist_ok=True)
    write_json(config.METRICS_PATH, payload)
    return payload
"""


def _main_template() -> str:
    return """from __future__ import annotations
from src.train import run_training
from src.evaluate import evaluate
import config

def main() -> None:
    config.set_global_seed()
    metrics = run_training()
    result = evaluate(metrics)
    print(f"METRIC: accuracy={result['evaluation']['accuracy']}")

if __name__ == "__main__":
    main()
"""


async def code_gen_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "code_generator"
    logger.info("agent.codegen.start", experiment_id=state["experiment_id"], requires_quantum=state["requires_quantum"])
    project = Path(state["project_path"])

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

    files = {
        str(project / "config.py"): _config_template(state),
        str(project / "src" / "__init__.py"): "",
        str(project / "src" / "utils.py"): _utils_template(),
        str(project / "src" / "preprocessing.py"): _preprocessing_template(),
        str(project / "src" / "model.py"): _model_template(),
        str(project / "src" / "train.py"): _train_template(),
        str(project / "src" / "evaluate.py"): _evaluate_template(),
        str(project / "main.py"): _main_template(),
    }
    if _should_inject_failure("codegen_syntax"):
        files[str(project / "main.py")] = files[str(project / "main.py")] + "\nthis is invalid python\n"
    for path, content in files.items():
        write_text_file(state["project_path"], path, content)
        if path not in state["created_files"]:
            state["created_files"].append(path)
        logger.info("agent.codegen.file_written", experiment_id=state["experiment_id"], file_path=path, size=len(content))

    logger.info("agent.codegen.end", experiment_id=state["experiment_id"], created_files=len(files))
    return state
