from __future__ import annotations

from src.prompts.clarifier import SYSTEM_PROMPT as CLARIFIER_PROMPT
from src.prompts.planner import SYSTEM_PROMPT as PLANNER_PROMPT
from src.prompts.env_manager import SYSTEM_PROMPT as ENV_PROMPT
from src.prompts.dataset_agent import SYSTEM_PROMPT as DATASET_PROMPT
from src.prompts.code_gen import SYSTEM_PROMPT as CODEGEN_PROMPT
from src.prompts.job_scheduler import SYSTEM_PROMPT as SCHEDULER_PROMPT
from src.prompts.error_recovery import SYSTEM_PROMPT as ERROR_PROMPT
from src.prompts.evaluator import SYSTEM_PROMPT as EVAL_PROMPT
from src.prompts.doc_generator import SYSTEM_PROMPT as DOC_PROMPT


PROMPT_BY_PHASE = {
    "clarifier": CLARIFIER_PROMPT,
    "planner": PLANNER_PROMPT,
    "env_manager": ENV_PROMPT,
    "dataset_manager": DATASET_PROMPT,
    "code_generator": CODEGEN_PROMPT,
    "job_scheduler": SCHEDULER_PROMPT,
    "error_recovery": ERROR_PROMPT,
    "results_evaluator": EVAL_PROMPT,
    "doc_generator": DOC_PROMPT,
}


def get_prompt_template(phase: str) -> str:
    return PROMPT_BY_PHASE.get(
        phase,
        "SYSTEM ROLE: General Agent\nRL POLICY:\n{rl_policy}\nCURRENT STATE:\n{state_json}",
    )

