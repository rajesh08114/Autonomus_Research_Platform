from __future__ import annotations

from src.state.research_state import ExperimentStatus, ResearchState
from src.core.logger import get_logger

logger = get_logger(__name__)


QUESTION_BANK = {
    "Q1": {"topic": "output_format", "text": "Do you want .py scripts or .ipynb notebooks?", "type": "choice", "options": [".py", ".ipynb"], "default": ".py"},
    "Q2": {"topic": "algorithm_class", "text": "What type of ML/AI task is this?", "type": "choice", "options": ["supervised", "unsupervised", "reinforcement", "quantum_ml"], "default": "supervised"},
    "Q3": {"topic": "requires_quantum", "text": "Does this experiment require quantum circuits?", "type": "boolean", "default": False},
    "Q4": {"topic": "quantum_framework", "text": "Which quantum framework should be used?", "type": "choice", "options": ["pennylane", "qiskit", "cirq", "no_preference"], "default": "pennylane"},
    "Q5": {"topic": "dataset_source", "text": "Where should the dataset come from?", "type": "choice", "options": ["kaggle", "sklearn", "synthetic", "upload"], "default": "sklearn"},
    "Q6": {"topic": "kaggle_dataset_id", "text": "Enter the Kaggle dataset path (username/dataset-name):", "type": "text", "default": None},
    "Q7": {"topic": "target_metric", "text": "What is the primary evaluation metric?", "type": "choice", "options": ["accuracy", "f1_macro", "rmse", "roc_auc", "fidelity"], "default": "accuracy"},
    "Q8": {"topic": "hardware_target", "text": "Target hardware for execution?", "type": "choice", "options": ["cpu", "cuda", "ibmq"], "default": "cpu"},
    "Q9": {"topic": "python_version", "text": "Which Python version?", "type": "choice", "options": ["3.10", "3.11", "3.12"], "default": "3.11"},
    "Q10": {"topic": "random_seed", "text": "Random seed for reproducibility?", "type": "number", "default": 42},
    "Q11": {"topic": "max_epochs", "text": "Maximum training epochs or circuit iterations?", "type": "number", "default": 50},
    "Q12": {"topic": "data_sensitivity", "text": "Is the dataset private or sensitive?", "type": "boolean", "default": False},
}

BASE_QUESTION_ORDER = ["Q1", "Q2", "Q3", "Q5", "Q7", "Q8", "Q10", "Q11"]
MAX_DYNAMIC_QUESTIONS = 12


def _infer_prompt_flags(prompt: str) -> dict[str, bool]:
    lower = prompt.lower()
    return {
        "requires_quantum_hint": any(token in lower for token in ["quantum", "vqe", "qaoa", "qnn", "circuit"]),
        "wants_notebook_hint": any(token in lower for token in ["notebook", "jupyter", ".ipynb"]),
        "wants_cuda_hint": "cuda" in lower,
    }


def _question_payload(qid: str, required: bool = True, default_override: object | None = None) -> dict:
    item = QUESTION_BANK[qid]
    payload = {
        "id": qid,
        "text": item["text"],
        "type": item["type"],
        "default": item["default"] if default_override is None else default_override,
        "required": required,
    }
    if "options" in item:
        payload["options"] = item["options"]
    return payload


def next_clarification_question_id(state: ResearchState, asked_question_ids: list[str]) -> str | None:
    asked = set(asked_question_ids)
    clar = state.get("clarifications", {})

    # Dynamic follow-ups based on previous answers.
    if bool(clar.get("requires_quantum")) and "Q4" not in asked:
        return "Q4"
    if str(clar.get("dataset_source", "")).strip().lower() == "kaggle" and "Q6" not in asked:
        return "Q6"

    for qid in BASE_QUESTION_ORDER:
        if qid not in asked:
            return qid
    return None


def build_clarification_question(
    qid: str,
    state: ResearchState,
    prompt_flags: dict[str, bool] | None = None,
) -> dict:
    flags = prompt_flags or _infer_prompt_flags(state["user_prompt"])
    default_override = None
    if qid == "Q1" and flags.get("wants_notebook_hint"):
        default_override = ".ipynb"
    if qid == "Q3" and flags.get("requires_quantum_hint"):
        default_override = True
    if qid == "Q8" and flags.get("wants_cuda_hint"):
        default_override = "cuda"

    return _question_payload(qid, required=qid not in {"Q10", "Q11"}, default_override=default_override)


async def clarifier_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "clarifier"
    logger.info("agent.clarifier.start", experiment_id=state["experiment_id"])
    flags = _infer_prompt_flags(state["user_prompt"])
    current_qid = next_clarification_question_id(state, asked_question_ids=[])
    if not current_qid:
        raise RuntimeError("Clarifier could not determine first question")
    current = build_clarification_question(current_qid, state, prompt_flags=flags)

    state["pending_user_question"] = {
        "mode": "sequential_dynamic",
        "current_question": current,
        "questions": [current],  # API contract: expose active question as a single-item list.
        "asked_question_ids": [],
        "answered": [],
        "answered_count": 0,
        "total_questions_planned": 0,
        "max_questions": MAX_DYNAMIC_QUESTIONS,
        "prompt_flags": flags,
    }
    state["status"] = ExperimentStatus.WAITING.value
    state["llm_calls_count"] += 1
    logger.info(
        "agent.clarifier.end",
        experiment_id=state["experiment_id"],
        active_question_id=current_qid,
        mode="sequential_dynamic",
    )
    return state
