from __future__ import annotations

import json
import time
from pathlib import Path

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode
from src.core.local_actions import queue_local_file_action
from src.core.logger import get_logger
from src.core.user_behavior import build_user_behavior_profile
from src.llm.master_llm import invoke_master_llm
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)

_REQUIRED_DOC_MARKERS = (
    "# Abstract",
    "## Research Objective",
    "## Experimental Results",
    "## Conclusion & Interpretation",
)
_EXPECTED_DOC_STRUCTURE = (
    "# Abstract",
    "## Research Objective",
    "## Methodology",
    "## Dataset Description",
    "## Experimental Results",
    "## Conclusion & Interpretation",
)


def _build_report(state: ResearchState) -> str:
    metrics = state.get("metrics", {})
    evaluation = metrics.get("evaluation", {})
    artifacts = metrics.get("artifacts", {}) if isinstance(metrics, dict) else {}
    preprocessing = metrics.get("preprocessing", {}) if isinstance(metrics, dict) else {}
    notebook_execution = metrics.get("notebook_execution", {}) if isinstance(metrics, dict) else {}
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
        "## Execution Trace Summary",
        json.dumps(
            [
                {
                    "script_path": item.get("script_path"),
                    "returncode": item.get("returncode"),
                    "duration_sec": item.get("duration_sec"),
                }
                for item in (state.get("execution_logs") or [])[-10:]
                if isinstance(item, dict)
            ],
            indent=2,
        ),
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
    if isinstance(notebook_execution, dict) and notebook_execution:
        sections.insert(16, "## Notebook Execution Summary\n" + json.dumps(notebook_execution, indent=2))
    if state["requires_quantum"]:
        sections.insert(12, "## Quantum Circuit Description\nSee `src/quantum_circuit.py` for generated circuit layer.")
    return "\n\n".join(sections)


def _missing_required_markers(markdown: str) -> list[str]:
    text = str(markdown or "").strip()
    missing: list[str] = []
    for marker in _REQUIRED_DOC_MARKERS:
        if marker not in text:
            missing.append(marker)
    return missing


def _looks_structured_report(markdown: str) -> bool:
    text = str(markdown or "")
    return all(section in text for section in _EXPECTED_DOC_STRUCTURE)


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
            "user_behavior_profile": build_user_behavior_profile(state),
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
    local_mode = is_vscode_execution_mode(state)
    docs_dir = Path(state["project_path"]) / "docs"
    report_path = docs_dir / "final_report.md"
    report_path_text = str(report_path)

    if local_mode:
        report_content = str(state.get("documentation_content") or "")
        if report_content and report_path_text in set(state.get("local_materialized_files", [])):
            state["documentation_path"] = report_path_text
            state["status"] = ExperimentStatus.SUCCESS.value
            state["phase"] = "finished"
            state["timestamp_end"] = time.time()
            logger.info(
                "agent.doc_generator.end",
                experiment_id=state["experiment_id"],
                report_path=report_path_text,
                local_materialized=True,
            )
            return state
    else:
        docs_dir.mkdir(parents=True, exist_ok=True)

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

    marker_patches = _missing_required_markers(report_markdown)
    if used_dynamic and (marker_patches or not _looks_structured_report(report_markdown)):
        logger.warning(
            "agent.doc.dynamic_structure_mismatch",
            experiment_id=state["experiment_id"],
            marker_patches=marker_patches,
        )
        report_markdown = _build_report(state)
        marker_patches = _missing_required_markers(report_markdown)
        fallback_static = True
        used_dynamic = False
    if marker_patches:
        logger.warning(
            "agent.doc.dynamic_validation_failed",
            experiment_id=state["experiment_id"],
            missing_markers=marker_patches,
        )
    state["documentation_path"] = report_path_text
    state["documentation_content"] = report_markdown
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
    summary = {
        "enabled": bool(settings.DOC_DYNAMIC_ENABLED),
        "used_dynamic": used_dynamic,
        "fallback_static": fallback_static,
        "marker_patches": marker_patches,
        "report_path": report_path_text,
        "pending_local_write": False,
    }
    state.setdefault("research_plan", {})["doc_generation_summary"] = summary

    if local_mode:
        state["local_file_plan"] = [
            item
            for item in state.get("local_file_plan", [])
            if str(item.get("path", "")) != report_path_text
        ]
        state["local_file_plan"].append({"path": report_path_text, "content": report_markdown, "phase": "doc_generator"})
        if report_path_text not in state["created_files"]:
            state["created_files"].append(report_path_text)
        queued = queue_local_file_action(
            state=state,
            phase="doc_generator",
            file_operations=[{"path": report_path_text, "content": report_markdown, "mode": "write", "phase": "doc_generator"}],
            next_phase="doc_generator",
            reason="Write final report locally to keep artifacts on the user machine",
            cwd=state["project_path"],
        )
        if queued:
            summary["pending_local_write"] = True
            logger.info(
                "agent.doc_generator.pending_local_action",
                experiment_id=state["experiment_id"],
                report_path=report_path_text,
            )
            return state
    else:
        report_path.write_text(report_markdown, encoding="utf-8")

    state["status"] = ExperimentStatus.SUCCESS.value
    state["phase"] = "finished"
    state["timestamp_end"] = time.time()
    logger.info("agent.doc_generator.end", experiment_id=state["experiment_id"], report_path=report_path_text)
    return state
