from __future__ import annotations

from pathlib import Path

from src.core.phase_validator import validate_phase_output
from src.state.research_state import ExperimentStatus, new_research_state


def _doc_markdown() -> str:
    return "\n\n".join(
        [
            "# Abstract",
            "summary",
            "## Research Objective",
            "objective",
            "## Experimental Results",
            "results",
            "## Conclusion & Interpretation",
            "conclusion",
        ]
    )


def test_doc_validator_accepts_pending_local_report_write(tmp_path: Path):
    state = new_research_state(
        "exp_doc_validate_local",
        str(tmp_path),
        "Generate report",
        {"execution_mode": "vscode_extension"},
    )
    report_path = str((tmp_path / "docs" / "final_report.md").resolve())
    report_markdown = _doc_markdown()
    state["phase"] = "doc_generator"
    state["status"] = ExperimentStatus.WAITING.value
    state["documentation_path"] = report_path
    state["documentation_content"] = report_markdown
    state["local_file_plan"] = [{"path": report_path, "content": report_markdown, "phase": "doc_generator"}]
    state["pending_user_confirm"] = {
        "action_id": "act_doc_1",
        "action": "apply_file_operations",
        "phase": "doc_generator",
        "next_phase": "doc_generator",
        "cwd": str(tmp_path),
        "file_operations": [{"path": report_path, "content": report_markdown, "mode": "write"}],
        "created_files": [report_path],
    }

    result = validate_phase_output("doc_generator", state)
    assert result.ok is True
    assert "final report file missing" not in result.errors
