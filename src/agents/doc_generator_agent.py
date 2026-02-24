from __future__ import annotations

import json
import time
from pathlib import Path

from src.core.logger import get_logger
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

def _build_report(state: ResearchState) -> str:
    metrics = state.get("metrics", {})
    evaluation = metrics.get("evaluation", {})
    sections = [
        "# Abstract",
        f"Experiment `{state['experiment_id']}` executed for objective: {state['user_prompt']}.",
        "## Research Objective",
        state["user_prompt"],
        "## Methodology",
        "\n".join(f"- {step}" for step in state["research_plan"].get("methodology", [])),
        "## Environment Setup",
        f"- Python: {state['python_version']}\n- Framework: {state['framework']}\n- Hardware: {state['hardware_target']}",
        "## Dataset Description",
        json.dumps(state.get("data_report", {}), indent=2),
        "## Model Architecture",
        state["research_plan"].get("algorithm", "N/A"),
        "## Training Configuration",
        f"- Seed: {state['random_seed']}\n- Epochs: {state.get('max_epochs')}\n- Batch size: {state.get('batch_size')}",
        "## Experimental Results",
        "| Metric | Value |\n|---|---|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in evaluation.items()),
        "## Error Incidents & Resolutions",
        json.dumps(state.get("repair_history", []), indent=2),
        "## Reproducibility Guide",
        "1. Install dependencies\n2. Run `python main.py`\n3. Compare outputs/metrics.json",
        "## Conclusion & Interpretation",
        f"Primary metric `{state['target_metric']}` = {evaluation.get(state['target_metric'], 'N/A')}.",
        "## Future Work",
        "1. Hyperparameter tuning\n2. Real-world dataset expansion\n3. Quantum backend benchmark",
        "## Appendix: Full State Snapshot",
        "```json\n" + json.dumps(state, indent=2, default=str) + "\n```",
    ]
    if state["requires_quantum"]:
        sections.insert(12, "## Quantum Circuit Description\nSee `src/quantum_circuit.py` for generated circuit layer.")
    return "\n\n".join(sections)


async def doc_generator_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "doc_generator"
    logger.info("agent.doc_generator.start", experiment_id=state["experiment_id"])
    docs_dir = Path(state["project_path"]) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "final_report.md"
    report_path.write_text(_build_report(state), encoding="utf-8")

    state["documentation_path"] = str(report_path)
    state["report_sections"] = [
        "abstract",
        "objective",
        "methodology",
        "environment_setup",
        "dataset_description",
        "model_architecture",
        "training_configuration",
        "experimental_results",
        "error_incidents",
        "reproducibility_guide",
        "conclusion",
        "future_work",
        "appendix",
    ]
    state["status"] = ExperimentStatus.SUCCESS.value
    state["phase"] = "finished"
    state["timestamp_end"] = time.time()
    logger.info("agent.doc_generator.end", experiment_id=state["experiment_id"], report_path=str(report_path))
    return state
