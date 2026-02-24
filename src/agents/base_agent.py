from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

from src.core.action_validator import validate_action
from src.core.rl_feedback import get_phase_policy_hint, record_phase_feedback, reward_from_validation
from src.core.state_compressor import compress_state
from src.llm.master_llm import invoke_master_llm
from src.llm.response_parser import parse_json_response
from src.prompts.registry import get_prompt_template
from src.state.research_state import ResearchState


class BaseAgent(ABC):
    MAX_RETRIES = 3

    @property
    @abstractmethod
    def phase_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def system_prompt_template(self) -> str:
        raise NotImplementedError

    async def invoke(self, state: ResearchState) -> ResearchState:
        start = time.time()
        state["phase"] = self.phase_name
        compressed = compress_state(state)
        rl_policy = await get_phase_policy_hint(self.phase_name)
        prompt_template = self.system_prompt_template or get_prompt_template(self.phase_name)
        prompt = (
            prompt_template.replace("{state_json}", json.dumps(compressed, indent=2, default=str))
            .replace("{rl_policy}", rl_policy)
        )
        action = await self._invoke_with_retry(prompt, state)
        state["llm_calls_count"] += 1
        state = await self.execute_action(action, state)
        await record_phase_feedback(
            experiment_id=state["experiment_id"],
            phase=self.phase_name,
            reward=reward_from_validation(valid=True, warning_count=0, error_count=0),
            signal="agent_action_valid",
            details={"action": action.get("action")},
        )
        state["phase_timings"][self.phase_name] = time.time() - start
        return state

    async def _invoke_with_retry(self, prompt: str, state: ResearchState) -> dict:
        for _ in range(self.MAX_RETRIES):
            raw = await invoke_master_llm(
                prompt,
                experiment_id=state["experiment_id"],
                phase=self.phase_name,
            )
            action = parse_json_response(raw)
            valid, _err = validate_action(action, state, phase=self.phase_name)
            if valid:
                return action
        raise RuntimeError(f"{self.phase_name} failed to produce valid action")

    @abstractmethod
    async def execute_action(self, action: dict, state: ResearchState) -> ResearchState:
        raise NotImplementedError
