from __future__ import annotations

import json
import re
from typing import Any

import pytest
import httpx

from src.api import app as api_app
from src.api.app import create_app
from src.agents import (
    base_agent,
    clarifier_agent,
    code_gen_agent,
    dataset_agent,
    doc_generator_agent,
    env_manager_agent,
    evaluator_agent,
    job_scheduler_agent,
    planner_agent,
    quantum_gate,
)
from src.config.settings import settings
from src.core import chat_assistant


def _extract_contract(user_prompt: str) -> tuple[str, str, str]:
    lowered = user_prompt.lower()
    problem = "classification"
    code_level = "intermediate"
    algorithm = "supervised"
    problem_match = re.search(r'"problem_type"\s*:\s*"([^"]+)"', lowered)
    level_match = re.search(r'"code_level"\s*:\s*"([^"]+)"', lowered)
    algo_match = re.search(r'"algorithm_class"\s*:\s*"([^"]+)"', lowered)
    if problem_match:
        problem = str(problem_match.group(1)).strip().lower()
    if level_match:
        code_level = str(level_match.group(1)).strip().lower()
    if algo_match:
        algorithm = str(algo_match.group(1)).strip().lower()
    return problem, code_level, algorithm


async def _fake_invoke_master_llm(
    system_prompt: str,
    user_prompt: str = "",
    experiment_id: str | None = None,
    phase: str | None = None,
) -> str:
    _ = (experiment_id, phase)
    context = f"{system_prompt}\n{user_prompt}".lower()

    if "parameters.questions" in context or "clarification agent" in context:
        payload = {
            "parameters": {
                "questions": [
                    {
                        "topic": "output_format",
                        "text": "Do you want .py scripts or .ipynb notebooks?",
                        "type": "choice",
                        "options": [".py", ".ipynb"],
                        "default": ".py",
                    },
                    {
                        "topic": "algorithm_class",
                        "text": "Which learning style should be prioritized?",
                        "type": "choice",
                        "options": ["supervised", "unsupervised", "reinforcement"],
                        "default": "supervised",
                    },
                ]
            }
        }
        return json.dumps(payload)

    if "system role: planner_dynamic_plan" in context:
        payload = {
            "objective_refinement": "Deliver a reproducible ML workflow aligned to user constraints.",
            "methodology": [
                "validate dataset quality and schema",
                "preprocess features and split data",
                "train model with deterministic seed",
                "evaluate primary and secondary metrics",
                "document results and reproducibility details",
            ],
            "algorithm": "state_aligned_model",
            "risk_checks": ["detect schema drift", "verify metric is numeric", "check reproducibility with fixed seed"],
            "package_additions": ["seaborn==0.13.2"],
            "training_strategy": "deterministic_baseline_then_refinement",
            "optimizer": "adam",
            "circuit_layers": 3,
            "encoding": "angle_encoding",
            "estimated_duration_minutes": 18,
        }
        return json.dumps(payload)

    if "system role: job_scheduler_dynamic_plan" in context:
        candidate = ""
        try:
            parsed = json.loads(user_prompt)
            candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            if isinstance(candidates, list) and candidates:
                candidate = str(candidates[0])
        except Exception:
            candidate = ""
        payload = {"execution_order": [candidate] if candidate else [], "rationale": "Run validation then main entrypoint order from state."}
        return json.dumps(payload)

    if "system role: quantum_gate_dynamic_code" in context:
        payload = {
            "code": (
                "from __future__ import annotations\n"
                "import pennylane as qml\n\n"
                "QUBIT_COUNT = 4\n"
                "CIRCUIT_LAYERS = 3\n"
                "BACKEND = 'default.qubit'\n\n"
                "class QuantumLayer:\n"
                "    def forward(self, x):\n"
                "        return [0.0]\n\n"
                "def get_circuit_diagram():\n"
                "    return 'ok'\n"
            ),
            "reasoning": "Minimal compliant quantum layer for tests.",
        }
        return json.dumps(payload)

    if "system role: evaluator_dynamic_interpretation" in context:
        payload = {
            "summary_text": "Metrics indicate the generated pipeline is stable for the requested task.",
            "insights": ["Primary metric is present and numeric.", "Training duration and artifacts were captured."],
            "warnings": ["Validate performance on an external holdout dataset."],
            "next_steps": ["Tune hyperparameters.", "Expand dataset coverage."],
        }
        return json.dumps(payload)

    if "system role: dataset_dynamic_plan" in context:
        payload = {
            "schema_mapping": {
                "feature_1": "feature_1",
                "feature_2": "feature_2",
                "feature_3": "feature_3",
                "target": "target",
            },
            "validation_checks": [
                {"type": "required_columns", "value": ["feature_1", "feature_2", "feature_3", "target"]},
                {"type": "min_rows", "value": 50},
                {"type": "target_binary", "value": True},
            ],
            "report_narrative": {
                "summary": "Dataset schema aligns with generated training pipeline contract.",
                "quality_notes": ["Validated row count and binary target consistency."],
            },
        }
        return json.dumps(payload)

    if "system role: env_dynamic_next_action" in context:
        package_spec = ""
        try:
            parsed = json.loads(user_prompt)
            candidates = parsed.get("candidate_specs", []) if isinstance(parsed, dict) else []
            if isinstance(candidates, list) and candidates:
                package_spec = str(candidates[0])
        except Exception:
            package_spec = ""
        payload = {"action": "install_package", "package_spec": package_spec, "reason": "Install next unresolved pinned dependency."}
        return json.dumps(payload)

    if "system role: doc_generation_markdown" in context:
        return (
            "# Abstract\n\n"
            "Dynamic report generated from experiment state.\n\n"
            "## Research Objective\n\n"
            "Produce a reproducible ML workflow from user constraints.\n\n"
            "## Methodology\n\n"
            "- Build deterministic dataset artifacts\n- Generate runnable code\n- Evaluate core metric\n\n"
            "## Experimental Results\n\n"
            "| Metric | Value |\n|---|---|\n| accuracy | 0.9 |\n\n"
            "## Conclusion & Interpretation\n\n"
            "The generated pipeline satisfies state constraints and produces valid outputs.\n"
        )

    if "codegen_file_manifest" in context:
        problem_type, code_level, algorithm_class = _extract_contract(user_prompt)
        repair_turn = "strict violations to fix exactly" in context
        utils_import_line = "import forbiddenlib\n" if not repair_turn else ""
        payload = {
            "problem_type": problem_type,
            "code_level": code_level,
            "algorithm_class": algorithm_class,
            "files": [
                {
                    "path": "config.py",
                    "content": (
                        "from __future__ import annotations\n"
                        "import os\n"
                        "import random\n\n"
                        "PROJECT_ROOT = os.path.dirname(__file__)\n"
                        "DATA_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')\n"
                        "DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')\n"
                        "MODEL_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'model_checkpoint')\n"
                        "METRICS_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'metrics.json')\n"
                        "PLOTS_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'plots')\n"
                        f"PROBLEM_TYPE = '{problem_type}'\n"
                        f"CODE_LEVEL = '{code_level}'\n"
                        f"ALGORITHM_CLASS = '{algorithm_class}'\n"
                        "TARGET_METRIC = 'accuracy'\n"
                        "RANDOM_SEED = 42\n"
                        "MAX_EPOCHS = 10\n"
                        "DEVICE = \"cpu\" if \"cpu\" != \"cuda\" else \"cuda\"\n\n"
                        "def set_global_seed() -> None:\n"
                        "    random.seed(RANDOM_SEED)\n"
                    ),
                },
                {"path": "src/__init__.py", "content": ""},
                {
                    "path": "src/utils.py",
                    "content": (
                        "from __future__ import annotations\n"
                        f"{utils_import_line}"
                        "import json\n"
                        "from pathlib import Path\n\n"
                        "def write_json(path: str, payload: dict) -> None:\n"
                        "    target = Path(path)\n"
                        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
                        "    target.write_text(json.dumps(payload, indent=2), encoding='utf-8')\n\n"
                        "def get_logger(name: str):\n"
                        "    import logging\n"
                        "    return logging.getLogger(name)\n"
                    ),
                },
                {
                    "path": "src/preprocessing.py",
                    "content": (
                        "from __future__ import annotations\n"
                        "import csv\n\n"
                        "def load_dataset(path: str) -> list[dict]:\n"
                        "    rows = []\n"
                        "    with open(path, 'r', encoding='utf-8') as handle:\n"
                        "        for row in csv.DictReader(handle):\n"
                        "            rows.append(dict(row))\n"
                        "    return rows\n\n"
                        "def preprocess_rows(rows: list[dict]) -> tuple[list[tuple[float, float, float]], list[float], dict]:\n"
                        "    features = []\n"
                        "    labels = []\n"
                        "    for row in rows:\n"
                        "        f1 = float(row.get('feature_1', 0.0))\n"
                        "        f2 = float(row.get('feature_2', 0.0))\n"
                        "        f3 = float(row.get('feature_3', 0.0))\n"
                        "        y = float(row.get('target', 0.0))\n"
                        "        features.append((f1, f2, f3))\n"
                        "        labels.append(y)\n"
                        "    return features, labels, {'rows': len(rows)}\n\n"
                        "def train_test_split(features: list, labels: list, seed: int = 42) -> tuple[list, list, list, list]:\n"
                        "    split = max(1, int(len(features) * 0.8))\n"
                        "    return features[:split], labels[:split], features[split:], labels[split:]\n"
                    ),
                },
                {
                    "path": "src/model.py",
                    "content": (
                        "from __future__ import annotations\n\n"
                        "class HybridResearchModel:\n"
                        "    def fit(self, X: list[tuple[float, float, float]], y: list[float]) -> None:\n"
                        "        self.threshold = 0.5\n\n"
                        "    def predict(self, X: list[tuple[float, float, float]]) -> list[float]:\n"
                        "        return [1.0 if (x[0] + x[1] + x[2]) / 3.0 > 0.5 else 0.0 for x in X]\n"
                    ),
                },
                {
                    "path": "src/train.py",
                    "content": (
                        "from __future__ import annotations\n"
                        "import os\n"
                        "import config\n"
                        "from src.preprocessing import load_dataset, preprocess_rows, train_test_split\n"
                        "from src.model import HybridResearchModel\n\n"
                        "def run_training() -> dict:\n"
                        "    rows = load_dataset(os.path.join(config.DATA_RAW_PATH, 'dataset.csv'))\n"
                        "    features, labels, prep = preprocess_rows(rows)\n"
                        "    train_x, train_y, test_x, test_y = train_test_split(features, labels, seed=config.RANDOM_SEED)\n"
                        "    model = HybridResearchModel()\n"
                        "    model.fit(train_x, train_y)\n"
                        "    preds = model.predict(test_x)\n"
                        "    correct = sum(1 for p, y in zip(preds, test_y) if int(p) == int(y))\n"
                        "    accuracy = correct / max(len(test_y), 1)\n"
                        "    return {\n"
                        "        'train_loss': float(max(0.0, 1.0 - accuracy)),\n"
                        "        'duration_sec': 0.1,\n"
                        "        'evaluation': {'accuracy': float(accuracy), 'f1_macro': float(accuracy), 'roc_auc': float(accuracy)},\n"
                        "        'preprocessing': prep,\n"
                        "        'training_history': [{'epoch': 1, 'loss': float(max(0.0, 1.0 - accuracy)), 'metric': float(accuracy)}],\n"
                        "        'plot_inputs': {'feature_1': [], 'feature_2': [], 'labels': []},\n"
                        "    }\n"
                    ),
                },
                {
                    "path": "src/evaluate.py",
                    "content": (
                        "from __future__ import annotations\n"
                        "import os\n"
                        "from src.utils import write_json\n"
                        "import config\n\n"
                        "def evaluate(metrics: dict) -> dict:\n"
                        "    evaluation = dict(metrics.get('evaluation', {}))\n"
                        "    primary = float(evaluation.get(config.TARGET_METRIC, 0.0))\n"
                        "    payload = {\n"
                        "        'experiment_id': 'test',\n"
                        "        'problem_type': config.PROBLEM_TYPE,\n"
                        "        'algorithm_class': config.ALGORITHM_CLASS,\n"
                        "        'algorithm': 'stub_algorithm',\n"
                        "        'framework': 'python',\n"
                        "        'target_metric': config.TARGET_METRIC,\n"
                        "        'primary_metric_name': config.TARGET_METRIC,\n"
                        "        'primary_metric_value': primary,\n"
                        "        'training': {'epochs': config.MAX_EPOCHS, 'final_loss': metrics.get('train_loss', 1.0), 'duration_sec': metrics.get('duration_sec', 0.0)},\n"
                        "        'evaluation': evaluation,\n"
                        "        'preprocessing': metrics.get('preprocessing', {}),\n"
                        "        'hardware': config.DEVICE,\n"
                        "        'seed': config.RANDOM_SEED,\n"
                        "        'reproducible': True,\n"
                        "        'quantum': {'enabled': False, 'framework': 'disabled', 'backend': 'n/a', 'qubits': 0},\n"
                        "        'artifacts': {'plots': [], 'preprocessing_report_path': os.path.join(config.DATA_PROCESSED_PATH, 'preprocessing_report.json'), 'training_history_path': os.path.join(config.DATA_PROCESSED_PATH, 'training_history.json'), 'explanation_path': ''},\n"
                        "    }\n"
                        "    write_json(config.METRICS_PATH, payload)\n"
                        "    return payload\n"
                    ),
                },
                {
                    "path": "main.py",
                    "content": (
                        "from __future__ import annotations\n"
                        "import config\n"
                        "from src.train import run_training\n"
                        "from src.evaluate import evaluate\n\n"
                        "def main() -> None:\n"
                        "    config.set_global_seed()\n"
                        "    metrics = run_training()\n"
                        "    result = evaluate(metrics)\n"
                        "    metric_name = result.get('primary_metric_name', config.TARGET_METRIC)\n"
                        "    metric_value = result.get('primary_metric_value', 0.0)\n"
                        "    print(f'METRIC: {metric_name}={metric_value}')\n\n"
                        "if __name__ == '__main__':\n"
                        "    main()\n"
                    ),
                },
            ],
        }
        return json.dumps(payload)

    if "methodology_additions" in context and "risk_checks" in context:
        payload = {
            "methodology_additions": ["calibrate decision threshold on validation data"],
            "risk_checks": ["verify train/test split reproducibility with fixed random seed"],
            "package_additions": ["seaborn==0.13.2"],
            "algorithm_override": "",
        }
        return json.dumps(payload)

    if "implementation_focus" in context and "failure_prevention" in context:
        payload = {
            "implementation_focus": "Build modular training and evaluation functions with explicit typed inputs.",
            "evaluation_focus": "Track primary metric per epoch and persist confusion diagnostics.",
            "failure_prevention": "Validate dataset schema and file paths before starting training.",
        }
        return json.dumps(payload)

    return json.dumps(
        {
            "action": "continue",
            "reasoning": "Unit-test LLM stub response",
            "parameters": {},
            "next_step": "planner",
            "confidence": 0.9,
        }
    )


async def _fake_chat_hf(
    question: str,
    selected: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
) -> dict[str, Any]:
    _ = (chat_history,)
    references = [str(item.get("experiment_id")) for item in selected if item.get("experiment_id")]
    answer = f"History-grounded response for: {question}. References: {', '.join(references[:3]) or 'none'}."
    prompt_tokens = max(1, int(len(question) / 4))
    completion_tokens = max(1, int(len(answer) / 4))
    return {
        "answer": answer,
        "follow_up_questions": [],
        "generation": {
            "provider": "huggingface",
            "model": settings.huggingface_model_id,
            "strategy": "history_grounded_chat_completion",
            "latency_ms": 1.0,
        },
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": 0.0,
        },
    }


async def _fake_llm_readiness() -> None:
    return None


@pytest.fixture(autouse=True)
def _stub_llm_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MASTER_LLM_PROVIDER", "huggingface")
    monkeypatch.setattr(settings, "HF_API_KEY", "test_hf_key")

    monkeypatch.setattr(base_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(clarifier_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(planner_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(code_gen_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(dataset_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(env_manager_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(job_scheduler_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(quantum_gate, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(evaluator_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(doc_generator_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(chat_assistant, "_invoke_huggingface_chat", _fake_chat_hf)
    monkeypatch.setattr(api_app, "assert_master_llm_ready", _fake_llm_readiness)


@pytest.fixture()
async def client() -> httpx.AsyncClient:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
