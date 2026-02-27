from __future__ import annotations

import json
import time
from pathlib import Path

from src.config.settings import settings
from src.core.logger import get_logger
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

_REQUIRED_DOC_MARKERS = (
    "# Abstract",
    "## Research Objective",
    "## Experimental Results",
    "## Conclusion & Interpretation",
)


def _build_report(state: ResearchState) -> str:
    metrics = state.get("metrics", {})
    evaluation = metrics.get("evaluation", {})
    artifacts = metrics.get("artifacts", {}) if isinstance(metrics, dict) else {}
    preprocessing = metrics.get("preprocessing", {}) if isinstance(metrics, dict) else {}
    research_plan = state.get("research_plan", {})
    problem_type = research_plan.get("problem_type", "classification")
    code_level = research_plan.get("code_level", "intermediate")
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
        "## User Requirement Mapping",
        f"- Problem type: {problem_type}\n- Code level: {code_level}\n- Research type: {state.get('research_type', 'ai')}",
        "## Training Configuration",
        f"- Seed: {state['random_seed']}\n- Epochs: {state.get('max_epochs')}\n- Batch size: {state.get('batch_size')}",
        "## Experimental Results",
        "| Metric | Value |\n|---|---|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in evaluation.items()),
        "## Data Preprocessing Summary",
        json.dumps(preprocessing, indent=2),
        "## Generated Plots & Artifacts",
        json.dumps(artifacts, indent=2),
        "## Error Incidents & Resolutions",
        json.dumps(state.get("repair_history", []), indent=2),
        "## Reproducibility Guide",
        "1. Install dependencies\n2. Run `python main.py`\n3. Compare outputs/metrics.json",
        "## Conclusion & Interpretation",
        f"Primary metric `{state['target_metric']}` = {evaluation.get(state['target_metric'], 'N/A')}.",
        "## LLM Usage Summary",
        f"- Total tokens: {state.get('total_tokens_used', 0)}\n- Cost controls: disabled",
        "## Future Work",
        "1. Hyperparameter tuning\n2. Real-world dataset expansion\n3. Quantum backend benchmark",
        "## Appendix: Full State Snapshot",
        "```json\n" + json.dumps(state, indent=2, default=str) + "\n```",
    ]
    if state["requires_quantum"]:
        sections.insert(12, "## Quantum Circuit Description\nSee `src/quantum_circuit.py` for generated circuit layer.")
    return "\n\n".join(sections)


def _ensure_required_markers(markdown: str) -> tuple[str, list[str]]:
    text = str(markdown or "").strip()
    if not text:
        text = ""
    added: list[str] = []
    for marker in _REQUIRED_DOC_MARKERS:
        if marker not in text:
            added.append(marker)
    if not added:
        return text, added
    if text:
        text = f"{text}\n\n"
    for marker in added:
        text += f"{marker}\n\n"
    return text.strip() + "\n", added


async def _generate_dynamic_report(state: ResearchState) -> str:
    system_prompt = (
        "SYSTEM ROLE: doc_generation_markdown.\n"
        "Generate a complete final research report in markdown from state context.\n"
        "You MUST include these exact headings:\n"
        "- # Abstract\n"
        "- ## Research Objective\n"
        "- ## Experimental Results\n"
        "- ## Conclusion & Interpretation\n"
        "Do not return JSON."
    )
    user_prompt = json.dumps(
        {
            "experiment_id": state.get("experiment_id"),
            "user_prompt": state.get("user_prompt"),
            "research_plan": state.get("research_plan", {}),
            "metrics": state.get("metrics", {}),
            "data_report": state.get("data_report", {}),
            "framework": state.get("framework"),
            "python_version": state.get("python_version"),
            "required_packages": state.get("required_packages", []),
            "repair_history": state.get("repair_history", []),
            "target_metric": state.get("target_metric"),
            "requires_quantum": state.get("requires_quantum"),
        },
        indent=2,
        default=str,
    )
    raw = await invoke_master_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        experiment_id=state["experiment_id"],
        phase="doc_generator",
    )
    state["llm_calls_count"] = int(state.get("llm_calls_count", 0)) + 1
    return str(raw or "").strip()


async def doc_generator_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "doc_generator"
    logger.info("agent.doc_generator.start", experiment_id=state["experiment_id"])
    docs_dir = Path(state["project_path"]) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "final_report.md"

    fallback_static = False
    used_dynamic = False
    marker_patches: list[str] = []
    report_markdown = ""

    if settings.DOC_DYNAMIC_ENABLED:
        try:
            report_markdown = await _generate_dynamic_report(state)
            if not report_markdown:
                raise RuntimeError("empty markdown response")
            used_dynamic = True
        except Exception as exc:
            logger.warning("agent.doc.dynamic_parse_failed", experiment_id=state["experiment_id"], error=str(exc))
            if settings.DYNAMIC_NONCODEGEN_FALLBACK_STATIC:
                fallback_static = True
                logger.warning("agent.doc.dynamic_fallback_static", experiment_id=state["experiment_id"], reason=str(exc))
                report_markdown = _build_report(state)
            else:
                raise
    else:
        fallback_static = True
        report_markdown = _build_report(state)

    report_markdown, marker_patches = _ensure_required_markers(report_markdown)
    if marker_patches:
        logger.warning(
            "agent.doc.dynamic_validation_failed",
            experiment_id=state["experiment_id"],
            missing_markers=marker_patches,
        )
    report_path.write_text(report_markdown, encoding="utf-8")

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
    state.setdefault("research_plan", {})["doc_generation_summary"] = {
        "enabled": bool(settings.DOC_DYNAMIC_ENABLED),
        "used_dynamic": used_dynamic,
        "fallback_static": fallback_static,
        "marker_patches": marker_patches,
        "report_path": str(report_path),
    }
    state["status"] = ExperimentStatus.SUCCESS.value
    state["phase"] = "finished"
    state["timestamp_end"] = time.time()
    logger.info("agent.doc_generator.end", experiment_id=state["experiment_id"], report_path=str(report_path))
    return state
