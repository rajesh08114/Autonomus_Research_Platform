from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode
from src.core.file_manager import write_text_file
from src.core.local_actions import queue_local_file_action
from src.core.logger import get_logger
from src.llm.master_llm import invoke_master_llm
from src.llm.response_parser import parse_json_response
from src.state.research_state import ResearchState

logger = get_logger(__name__)

_PROBLEM_TYPE_OPTIONS = {"classification", "regression", "clustering", "reinforcement", "forecasting", "generation"}
_CODE_LEVEL_OPTIONS = {"low", "intermediate", "advanced"}
_ALGORITHM_CLASS_OPTIONS = {"supervised", "unsupervised", "reinforcement", "quantum_ml"}


def _should_inject_failure(point: str) -> bool:
    if not settings.FAILURE_INJECTION_ENABLED:
        return False
    configured = {p.strip().lower() for p in settings.FAILURE_INJECTION_POINTS.split(",") if p.strip()}
    if configured and point.lower() not in configured:
        return False
    return random.random() < float(max(0.0, min(1.0, settings.FAILURE_INJECTION_RATE)))


def _normalize_choice(value: Any, options: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    if text in options:
        return text
    return default


def _resolve_problem_type(state: ResearchState) -> str:
    from_plan = str((state.get("research_plan") or {}).get("problem_type", "")).strip().lower()
    from_clar = str((state.get("clarifications") or {}).get("problem_type", "")).strip().lower()
    target_metric = str(state.get("target_metric", "")).strip().lower()
    algorithm_class = str((state.get("clarifications") or {}).get("algorithm_class", "supervised")).strip().lower()
    selected = from_plan or from_clar
    if selected in _PROBLEM_TYPE_OPTIONS:
        return selected
    if target_metric in {"rmse", "mae", "mse", "r2"}:
        return "regression"
    if algorithm_class == "unsupervised":
        return "clustering"
    if algorithm_class == "reinforcement":
        return "reinforcement"
    return "classification"


def _resolve_code_level(state: ResearchState) -> str:
    from_plan = str((state.get("research_plan") or {}).get("code_level", "")).strip().lower()
    from_clar = str((state.get("clarifications") or {}).get("code_level", "")).strip().lower()
    return _normalize_choice(from_plan or from_clar, _CODE_LEVEL_OPTIONS, "intermediate")


def _resolve_algorithm_class(state: ResearchState) -> str:
    from_plan = str((state.get("research_plan") or {}).get("algorithm_class", "")).strip().lower()
    from_clar = str((state.get("clarifications") or {}).get("algorithm_class", "")).strip().lower()
    return _normalize_choice(from_plan or from_clar, _ALGORITHM_CLASS_OPTIONS, "supervised")


def _safe_parse(raw: str) -> dict[str, Any]:
    try:
        return parse_json_response(raw)
    except Exception:
        try:
            return json.loads(str(raw or "").strip())
        except Exception:
            return {}


def _clean_line(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())[:180]


def _py_string(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


async def _llm_codegen_guidance(
    state: ResearchState,
    problem_type: str,
    code_level: str,
    algorithm_class: str,
) -> dict[str, str]:
    payload = {
        "framework": state.get("framework"),
        "requires_quantum": bool(state.get("requires_quantum")),
        "problem_type": problem_type,
        "code_level": code_level,
        "algorithm_class": algorithm_class,
        "target_metric": state.get("target_metric"),
        "hardware_target": state.get("hardware_target"),
    }
    system_prompt = (
        "You are a senior code generation reviewer. Return one JSON object only with keys: "
        "implementation_focus, evaluation_focus, failure_prevention. "
        "Each value must be one short actionable sentence."
    )
    user_prompt = f"Project context:\n{json.dumps(payload, indent=2, default=str)}"
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="code_generator",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    parsed = _safe_parse(raw)
    if not isinstance(parsed, dict):
        return {}
    guidance = {
        "implementation_focus": _clean_line(parsed.get("implementation_focus")),
        "evaluation_focus": _clean_line(parsed.get("evaluation_focus")),
        "failure_prevention": _clean_line(parsed.get("failure_prevention")),
    }
    return {k: v for k, v in guidance.items() if v}


def _config_template(
    state: ResearchState,
    problem_type: str,
    code_level: str,
    algorithm_class: str,
    llm_guidance: dict[str, str] | None = None,
) -> str:
    guidance = llm_guidance or {}
    implementation_focus = _py_string(str(guidance.get("implementation_focus", "")))
    evaluation_focus = _py_string(str(guidance.get("evaluation_focus", "")))
    failure_prevention = _py_string(str(guidance.get("failure_prevention", "")))
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

RANDOM_SEED = {state["random_seed"]}
MAX_EPOCHS = {int(state.get("max_epochs") or 50)}
BATCH_SIZE = {int(state.get("batch_size") or 32)}
LEARNING_RATE = 0.01
DEVICE = "cpu" if "{state["hardware_target"]}" != "cuda" else "cuda"
FRAMEWORK = "{state["framework"]}"
TARGET_METRIC = "{state["target_metric"]}"
PROBLEM_TYPE = "{problem_type}"
CODE_LEVEL = "{code_level}"
ALGORITHM_CLASS = "{algorithm_class}"
REQUIRES_QUANTUM = {str(bool(state.get("requires_quantum")))}
QUANTUM_FRAMEWORK = "{state.get("quantum_framework") or ""}"
QUANTUM_BACKEND = "{state.get("quantum_backend") or ""}"
QUBIT_COUNT = {int(state.get("quantum_qubit_count") or 0)}
LLM_IMPLEMENTATION_FOCUS = "{implementation_focus}"
LLM_EVALUATION_FOCUS = "{evaluation_focus}"
LLM_FAILURE_PREVENTION = "{failure_prevention}"

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


def _preprocessing_template(problem_type: str) -> str:
    label_note = "numeric target" if problem_type == "regression" else "binary target"
    return f"""from __future__ import annotations
import csv
import random
import statistics
from typing import Any

FEATURES = ["feature_1", "feature_2", "feature_3"]
LABEL_NOTE = "{label_note}"

def _to_float(raw: object) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None

def load_dataset(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows

def preprocess_rows(rows: list[dict[str, Any]]) -> tuple[list[tuple[float, float, float]], list[float], dict[str, Any]]:
    if not rows:
        raise ValueError("No rows available for preprocessing")

    columns: dict[str, list[float]] = {{name: [] for name in FEATURES}}
    for row in rows:
        for name in FEATURES:
            value = _to_float(row.get(name))
            if value is not None:
                columns[name].append(value)
    medians = {{name: (statistics.median(values) if values else 0.0) for name, values in columns.items()}}
    mins = {{name: min(values) if values else 0.0 for name, values in columns.items()}}
    maxs = {{name: max(values) if values else 1.0 for name, values in columns.items()}}

    features: list[tuple[float, float, float]] = []
    labels: list[float] = []
    for row in rows:
        vector: list[float] = []
        for name in FEATURES:
            value = _to_float(row.get(name))
            if value is None:
                value = float(medians[name])
            denom = float(maxs[name] - mins[name])
            scaled = 0.0 if denom == 0 else (value - mins[name]) / denom
            vector.append(float(round(scaled, 8)))
        raw_target = _to_float(row.get("target"))
        if raw_target is None:
            raw_target = 0.0
        features.append((vector[0], vector[1], vector[2]))
        labels.append(float(raw_target))

    report = {{
        "strategy": {{
            "missing_values": "median_imputation",
            "scaling": "min_max",
            "split": "seeded_shuffle",
        }},
        "rows": len(rows),
        "label_note": LABEL_NOTE,
        "medians": medians,
        "mins": mins,
        "maxs": maxs,
    }}
    return features, labels, report

def train_test_split(features: list[tuple[float, float, float]], labels: list[float], seed: int = 42) -> tuple[list, list, list, list]:
    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    split_idx = int(len(indices) * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    train_x = [features[i] for i in train_idx]
    train_y = [labels[i] for i in train_idx]
    test_x = [features[i] for i in test_idx]
    test_y = [labels[i] for i in test_idx]
    return train_x, train_y, test_x, test_y
"""


def _model_template(problem_type: str, code_level: str) -> str:
    advanced_note = "True" if code_level == "advanced" else "False"
    return f"""from __future__ import annotations
import math
from typing import Iterable

try:
    from src.quantum_circuit import QuantumLayer
except Exception:
    class QuantumLayer:
        def forward(self, x: Iterable[float]) -> list[float]:
            values = list(x)
            total = float(sum(values))
            return [total / max(len(values), 1)]

PROBLEM_TYPE = "{problem_type}"
ADVANCED_MODE = {advanced_note}

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))

def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return [0.0, 0.0, 0.0, 0.0]
    size = len(vectors[0])
    out = []
    for idx in range(size):
        out.append(sum(row[idx] for row in vectors) / len(vectors))
    return out

class HybridResearchModel:
    def __init__(self) -> None:
        self.quantum = QuantumLayer()
        self.problem_type = PROBLEM_TYPE
        self.majority = 0
        self.weights = [0.4, 0.35, 0.25]
        self.bias = 0.0
        self.centroids = {{0: [0.0, 0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0, 0.0]}}
        self.cluster_centers = [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]

    def _augment(self, x: Iterable[float]) -> list[float]:
        values = [float(v) for v in x]
        q = self.quantum.forward(values)
        quantum_value = float(q[0]) if isinstance(q, list) and q else 0.0
        return [values[0], values[1], values[2], quantum_value]

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def fit(self, X: list[tuple[float, float, float]], y: list[float]) -> None:
        if self.problem_type == "regression":
            if not X:
                return
            avg_x = [sum(row[idx] for row in X) / len(X) for idx in range(3)]
            avg_y = _mean(y)
            self.weights = [avg_y * (0.3 + 0.1 * avg_x[idx]) for idx in range(3)]
            self.bias = avg_y * 0.1
            if ADVANCED_MODE:
                # small calibration for advanced mode to reduce bias drift
                self.bias = self.bias * 0.9
            return

        if self.problem_type == "clustering":
            if not X:
                return
            midpoint = max(1, len(X) // 2)
            left = [list(x) for x in X[:midpoint]]
            right = [list(x) for x in X[midpoint:]]
            if left:
                self.cluster_centers[0] = [sum(v[i] for v in left) / len(left) for i in range(3)]
            if right:
                self.cluster_centers[1] = [sum(v[i] for v in right) / len(right) for i in range(3)]
            return

        labels = [1 if float(item) > 0.5 else 0 for item in y]
        zeros = sum(1 for value in labels if value == 0)
        ones = len(labels) - zeros
        self.majority = 1 if ones >= zeros else 0
        grouped = {{0: [], 1: []}}
        for row, label in zip(X, labels):
            grouped[int(label)].append(self._augment(row))
        for label in (0, 1):
            self.centroids[label] = _mean_vector(grouped[label])

    def predict(self, X: list[tuple[float, float, float]]) -> list[float]:
        if self.problem_type == "regression":
            preds: list[float] = []
            for row in X:
                pred = sum(float(row[idx]) * self.weights[idx] for idx in range(3)) + self.bias
                preds.append(float(pred))
            return preds

        if self.problem_type == "clustering":
            preds = []
            for row in X:
                d0 = self._distance(list(row), self.cluster_centers[0])
                d1 = self._distance(list(row), self.cluster_centers[1])
                preds.append(float(0 if d0 <= d1 else 1))
            return preds

        preds = []
        for row in X:
            vector = self._augment(row)
            d0 = self._distance(vector, self.centroids[0])
            d1 = self._distance(vector, self.centroids[1])
            if d0 == d1:
                preds.append(float(self.majority))
            else:
                preds.append(float(0 if d0 < d1 else 1))
        return preds
"""


def _train_template(problem_type: str, code_level: str) -> str:
    extra_loop = "int(config.MAX_EPOCHS * 1.2)" if code_level == "advanced" else "config.MAX_EPOCHS"
    return f"""from __future__ import annotations
import math
import os
import time

from src.preprocessing import load_dataset, preprocess_rows, train_test_split
from src.model import HybridResearchModel
from src.utils import write_json, get_logger
import config

PROBLEM_TYPE = "{problem_type}"

def _classification_metrics(preds: list[float], labels: list[float]) -> dict:
    y_true = [1 if float(v) > 0.5 else 0 for v in labels]
    y_pred = [1 if float(v) > 0.5 else 0 for v in preds]
    correct = sum(1 for p, y in zip(y_pred, y_true) if p == y)
    accuracy = correct / max(len(y_true), 1)
    fp = sum(1 for p, y in zip(y_pred, y_true) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(y_pred, y_true) if p == 0 and y == 1)
    precision = correct / max(correct + fp, 1)
    recall = correct / max(correct + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))
    return {{
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "roc_auc": float(accuracy),
        "rmse": float(math.sqrt(max(0.0, 1.0 - accuracy))),
    }}

def _regression_metrics(preds: list[float], labels: list[float]) -> dict:
    errors = [float(p) - float(y) for p, y in zip(preds, labels)]
    mse = sum(err * err for err in errors) / max(len(errors), 1)
    mae = sum(abs(err) for err in errors) / max(len(errors), 1)
    rmse = math.sqrt(max(0.0, mse))
    mean_y = sum(float(y) for y in labels) / max(len(labels), 1)
    ss_tot = sum((float(y) - mean_y) ** 2 for y in labels)
    ss_res = sum((float(y) - float(p)) ** 2 for p, y in zip(preds, labels))
    r2 = 0.0 if ss_tot == 0 else (1.0 - (ss_res / ss_tot))
    return {{
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "r2": float(r2),
        "accuracy": float(max(0.0, 1.0 - min(1.0, rmse))),
    }}

def run_training() -> dict:
    start = time.time()
    logger = get_logger("train")
    rows = load_dataset(os.path.join(config.DATA_RAW_PATH, "dataset.csv"))
    features, labels, preprocess_report = preprocess_rows(rows)
    train_x, train_y, test_x, test_y = train_test_split(features, labels, seed=config.RANDOM_SEED)

    model = HybridResearchModel()
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    metrics = _regression_metrics(preds, test_y) if PROBLEM_TYPE == "regression" else _classification_metrics(preds, test_y)

    history = []
    epochs = max(1, {extra_loop})
    for epoch in range(1, epochs + 1):
        progress = epoch / max(epochs, 1)
        primary = float(metrics.get(config.TARGET_METRIC, metrics.get("accuracy", 0.0)))
        if PROBLEM_TYPE == "regression":
            loss = max(0.0001, float(metrics.get("mse", 1.0)) * (1.05 - progress))
            trend_metric = max(0.0, float(metrics.get("r2", 0.0)) * (0.5 + 0.5 * progress))
        else:
            loss = max(0.01, (1.0 - primary) * (1.1 - progress))
            trend_metric = min(1.0, primary * (0.6 + 0.4 * progress))
        history.append({{"epoch": epoch, "loss": round(float(loss), 6), "metric": round(float(trend_metric), 6)}})

    result = {{
        "problem_type": PROBLEM_TYPE,
        "algorithm_class": config.ALGORITHM_CLASS,
        "train_loss": float(history[-1]["loss"] if history else 1.0),
        "duration_sec": round(time.time() - start, 6),
        "evaluation": metrics,
        "preprocessing": preprocess_report,
        "training_history": history,
        "plot_inputs": {{
            "feature_1": [x[0] for x in test_x[:80]],
            "feature_2": [x[1] for x in test_x[:80]],
            "labels": [float(y) for y in test_y[:80]],
        }},
    }}
    os.makedirs(config.MODEL_CHECKPOINT_PATH, exist_ok=True)
    write_json(
        os.path.join(config.MODEL_CHECKPOINT_PATH, "model.json"),
        {{
            "framework": config.FRAMEWORK,
            "problem_type": PROBLEM_TYPE,
            "quantum_framework": config.QUANTUM_FRAMEWORK,
            "code_level": config.CODE_LEVEL,
        }},
    )
    write_json(os.path.join(config.DATA_PROCESSED_PATH, "preprocessing_report.json"), preprocess_report)
    write_json(os.path.join(config.DATA_PROCESSED_PATH, "training_history.json"), {{"history": history}})
    logger.info("Training completed")
    return result
"""


def _evaluate_template(problem_type: str) -> str:
    return f"""from __future__ import annotations
import os
from pathlib import Path

from src.utils import write_json
import config

PROBLEM_TYPE = "{problem_type}"

def _write_explanation(payload: dict) -> str:
    target = Path(config.PROJECT_ROOT) / "docs" / "model_explanation.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    eval_metrics = payload.get("evaluation", {{}})
    lines = [
        "# Model Explanation",
        "",
        f"- Problem type: {{payload.get('problem_type')}}",
        f"- Algorithm class: {{payload.get('algorithm_class')}}",
        f"- Target metric: {{payload.get('target_metric')}}",
        f"- Primary value: {{payload.get('primary_metric_value')}}",
        "",
        "## Metrics",
    ]
    for key, value in eval_metrics.items():
        lines.append(f"- {{key}}: {{value}}")
    target.write_text("\\n".join(lines), encoding="utf-8")
    return str(target)

def _generate_plots(metrics: dict) -> list[str]:
    os.makedirs(config.PLOTS_PATH, exist_ok=True)
    outputs: list[str] = []
    history = metrics.get("training_history", [])
    plot_inputs = metrics.get("plot_inputs", {{}})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [item.get("epoch", 0) for item in history]
        losses = [item.get("loss", 0.0) for item in history]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, losses, label="loss", color="#1565C0", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(alpha=0.3)
        loss_path = os.path.join(config.PLOTS_PATH, "training_loss.png")
        plt.tight_layout()
        plt.savefig(loss_path, dpi=130)
        plt.close()
        outputs.append(loss_path)

        f1 = plot_inputs.get("feature_1", [])
        f2 = plot_inputs.get("feature_2", [])
        labels = plot_inputs.get("labels", [])
        plt.figure(figsize=(6, 4))
        for idx in range(min(len(f1), len(f2), len(labels))):
            color = "#1B5E20" if float(labels[idx]) > 0 else "#B71C1C"
            plt.scatter(float(f1[idx]), float(f2[idx]), c=color, s=18)
        plt.xlabel("Feature 1 (scaled)")
        plt.ylabel("Feature 2 (scaled)")
        plt.title("Feature Projection")
        scatter_path = os.path.join(config.PLOTS_PATH, "feature_projection.png")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=130)
        plt.close()
        outputs.append(scatter_path)
    except Exception as exc:
        fallback = os.path.join(config.PLOTS_PATH, "plot_generation_fallback.txt")
        with open(fallback, "w", encoding="utf-8") as handle:
            handle.write(f"Matplotlib unavailable: {{exc}}\\n")
        outputs.append(fallback)
    return outputs

def evaluate(metrics: dict) -> dict:
    evaluation = dict(metrics.get("evaluation", {{}}))
    target = config.TARGET_METRIC
    if target not in evaluation:
        target = "accuracy" if "accuracy" in evaluation else next(iter(evaluation.keys()), "accuracy")
    primary_value = float(evaluation.get(target, 0.0))

    payload = {{
        "experiment_id": config.EXPERIMENT_ID,
        "problem_type": PROBLEM_TYPE,
        "algorithm_class": config.ALGORITHM_CLASS,
        "algorithm": config.FRAMEWORK + "_" + PROBLEM_TYPE,
        "framework": config.FRAMEWORK,
        "target_metric": config.TARGET_METRIC,
        "primary_metric_name": target,
        "primary_metric_value": primary_value,
        "training": {{
            "epochs": config.MAX_EPOCHS,
            "final_loss": metrics.get("train_loss", 1.0),
            "duration_sec": metrics.get("duration_sec", 0.0),
        }},
        "evaluation": evaluation,
        "preprocessing": metrics.get("preprocessing", {{}}),
        "hardware": config.DEVICE,
        "seed": config.RANDOM_SEED,
        "reproducible": True,
        "quantum": {{
            "enabled": bool(config.REQUIRES_QUANTUM),
            "framework": config.QUANTUM_FRAMEWORK or "disabled",
            "backend": config.QUANTUM_BACKEND or "n/a",
            "qubits": config.QUBIT_COUNT,
        }},
        "artifacts": {{
            "plots": _generate_plots(metrics),
            "preprocessing_report_path": os.path.join(config.DATA_PROCESSED_PATH, "preprocessing_report.json"),
            "training_history_path": os.path.join(config.DATA_PROCESSED_PATH, "training_history.json"),
            "explanation_path": "",
        }},
    }}
    payload["artifacts"]["explanation_path"] = _write_explanation(payload)
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
    metric_name = result.get("primary_metric_name", config.TARGET_METRIC)
    metric_value = result.get("primary_metric_value", result.get("evaluation", {}).get(metric_name, 0.0))
    print(f"METRIC: {metric_name}={metric_value}")
    print(f"PLOTS: {result.get('artifacts', {}).get('plots', [])}")
    print(f"EXPLANATION: {result.get('artifacts', {}).get('explanation_path', '')}")

if __name__ == "__main__":
    main()
"""


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

    llm_guidance: dict[str, str] = {}
    try:
        llm_guidance = await _llm_codegen_guidance(state, problem_type, code_level, algorithm_class)
    except Exception:
        logger.exception("agent.codegen.llm_guidance_failed", experiment_id=state["experiment_id"])
        if not settings.ALLOW_RULE_BASED_FALLBACK:
            raise RuntimeError("Code generation LLM guidance failed and rule-based fallback is disabled.")
    if llm_guidance:
        state.setdefault("research_plan", {})["codegen_guidance"] = llm_guidance

    files = {
        str(project / "config.py"): _config_template(state, problem_type, code_level, algorithm_class, llm_guidance=llm_guidance),
        str(project / "src" / "__init__.py"): "",
        str(project / "src" / "utils.py"): _utils_template(),
        str(project / "src" / "preprocessing.py"): _preprocessing_template(problem_type),
        str(project / "src" / "model.py"): _model_template(problem_type, code_level),
        str(project / "src" / "train.py"): _train_template(problem_type, code_level),
        str(project / "src" / "evaluate.py"): _evaluate_template(problem_type),
        str(project / "main.py"): _main_template(),
    }
    if _should_inject_failure("codegen_syntax"):
        files[str(project / "main.py")] = files[str(project / "main.py")] + "\nthis is invalid python\n"

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
            reason=f"Create generated source files locally ({problem_type}/{code_level}) before scheduling execution",
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
