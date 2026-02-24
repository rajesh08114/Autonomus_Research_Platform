from __future__ import annotations

import random
import string
import time

from src.core.logger import get_logger
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

def _package_set_for_state(state: ResearchState) -> list[str]:
    packages = {
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib==3.8.4",
        "scikit-learn==1.4.2",
        "structlog==24.1.0",
    }
    framework = state["framework"]
    if framework == "pytorch":
        packages.update({"torch==2.2.0", "torchvision==0.17.0"})
    if state["requires_quantum"]:
        qf = state["quantum_framework"] or "pennylane"
        if qf == "pennylane":
            packages.update({"pennylane==0.36.0", "pennylane-lightning==0.36.0"})
        elif qf == "qiskit":
            packages.update({"qiskit==1.1.0", "qiskit-aer==0.14.0"})
        elif qf == "cirq":
            packages.add("cirq==1.3.0")
    if state["dataset_source"] == "kaggle":
        packages.add("kaggle==1.6.12")
    if state["output_format"] == ".ipynb":
        packages.update({"jupyter==1.0.0", "ipykernel==6.29.5"})
    return sorted(packages)


def _new_project_id() -> str:
    date = time.strftime("%Y%m%d")
    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"exp_{date}_{suffix}"


async def planner_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "planner"
    logger.info("agent.planner.start", experiment_id=state["experiment_id"])
    clar = state["clarifications"]

    algorithm_class = str(clar.get("algorithm_class", "supervised"))
    state["requires_quantum"] = bool(clar.get("requires_quantum", False))
    state["quantum_framework"] = None if clar.get("quantum_framework") == "no_preference" else clar.get("quantum_framework")
    state["dataset_source"] = str(clar.get("dataset_source", "sklearn"))
    state["output_format"] = str(clar.get("output_format", ".py"))
    state["target_metric"] = str(clar.get("target_metric", "accuracy"))
    state["hardware_target"] = str(clar.get("hardware_target", "cpu"))
    state["random_seed"] = int(clar.get("random_seed", 42))
    state["max_epochs"] = int(clar.get("max_epochs", 50))
    state["python_version"] = str(clar.get("python_version", "3.11"))
    state["kaggle_dataset_id"] = clar.get("kaggle_dataset_id")

    if state["requires_quantum"]:
        state["framework"] = "pytorch"
        state["quantum_algorithm"] = "VQE"
        state["quantum_qubit_count"] = 4
        state["quantum_backend"] = "default.qubit"
    elif algorithm_class == "reinforcement":
        state["framework"] = "pytorch"
    else:
        state["framework"] = "sklearn"

    state["research_plan"] = {
        "objective": state["user_prompt"],
        "methodology": [
            "collect or generate dataset",
            "preprocess and split data",
            "train baseline model",
            "evaluate target metric",
            "document artifacts",
        ],
        "algorithm": "quantum_hybrid_classifier" if state["requires_quantum"] else "classical_classifier",
        "framework": state["framework"],
        "dataset": {
            "source": state["dataset_source"],
            "kaggle_dataset_id": state["kaggle_dataset_id"],
            "expected_shape": "tabular rows x features+target",
        },
        "metrics": [state["target_metric"], "train_loss", "duration_sec"],
        "hardware": {
            "target": state["hardware_target"],
            "fallback": "cpu",
        },
        "reproducibility": {
            "seed": state["random_seed"],
            "pins": True,
        },
        "estimated_duration_minutes": 15,
    }
    state["required_packages"] = _package_set_for_state(state)
    state["status"] = ExperimentStatus.RUNNING.value
    logger.info(
        "agent.planner.end",
        experiment_id=state["experiment_id"],
        framework=state["framework"],
        requires_quantum=state["requires_quantum"],
        package_count=len(state["required_packages"]),
    )
    return state
