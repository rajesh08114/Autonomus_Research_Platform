from __future__ import annotations

import pytest

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
