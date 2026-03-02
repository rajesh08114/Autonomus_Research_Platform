from __future__ import annotations

import pytest

from src.agents import clarifier_agent
from src.agents.clarifier_agent import clarifier_agent_node
from src.state.research_state import new_research_state


@pytest.mark.asyncio
async def test_clarifier_generates_questions(tmp_path):
    state = new_research_state("exp_1", str(tmp_path), "build quantum classifier", {})
    state = await clarifier_agent_node(state)
    assert state["pending_user_question"] is not None
    assert len(state["pending_user_question"]["questions"]) == 1
    assert state["pending_user_question"]["current_question"]["id"] == state["pending_user_question"]["questions"][0]["id"]
    assert state["pending_user_question"]["current_question"]["topic"]


@pytest.mark.asyncio
async def test_clarifier_accepts_fenced_json(monkeypatch, tmp_path):
    async def _fenced_llm(*args, **kwargs):
        return """```json
{
  "parameters": {
    "questions": [
      {
        "topic": "output_format",
        "text": "Choose output format",
        "type": "choice",
        "options": [".py", ".ipynb"],
        "required": true
      }
    ]
  }
}
```"""

    monkeypatch.setattr(clarifier_agent, "invoke_master_llm", _fenced_llm)
    state = new_research_state("exp_fenced", str(tmp_path), "build classifier", {})
    state = await clarifier_agent_node(state)

    pending = state.get("pending_user_question") or {}
    current = pending.get("current_question") or {}
    assert current.get("topic") == "output_format"
    assert current.get("type") == "choice"


@pytest.mark.asyncio
async def test_clarifier_caps_question_plan_to_max(monkeypatch, tmp_path):
    many_questions = [
        {
            "topic": f"topic_{i}",
            "text": f"Question {i}",
            "type": "text",
            "required": True,
        }
        for i in range(1, 13)
    ]

    async def _many_llm_json(*args, **kwargs):
        import json
        return json.dumps({"parameters": {"questions": many_questions}})

    monkeypatch.setattr(clarifier_agent, "invoke_master_llm", _many_llm_json)
    state = new_research_state("exp_cap", str(tmp_path), "build robust classifier", {})
    state = await clarifier_agent_node(state)

    pending = state.get("pending_user_question") or {}
    plan = pending.get("question_plan") or []
    assert len(plan) <= clarifier_agent.MAX_DYNAMIC_QUESTIONS
    assert pending.get("total_questions_planned") <= clarifier_agent.MAX_DYNAMIC_QUESTIONS
