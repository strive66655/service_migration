from __future__ import annotations

from dataclasses import replace
from typing import Optional

from src.algorithms.base_policy import BasePolicy
from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import User
from src.env.mec_env import MECEnvironment
from src.llm_core.llm_client import BaseLLMClient, RuleBasedLLMClient, build_llm_client
from src.llm_core.parser import parse_llm_response
from src.llm_core.prompt_builder import build_llm_prompt


class LLMPolicy(BasePolicy):
    def __init__(
        self,
        base_params: PolicyParams,
        refresh_interval: int = 1,
        retry_remote_every: int = 5,
    ) -> None:
        self.base_params = base_params
        self.active_params = base_params
        self.refresh_interval = max(1, refresh_interval)
        self.retry_remote_every = max(1, retry_remote_every)
        self.last_refresh_step = -1
        self.last_failure_step = -1
        self.last_rationale = "fallback to base parameters"
        self.last_error = ""
        self.client: BaseLLMClient = build_llm_client()
        self.executor = CostAwarePolicy(self.active_params)

    def _should_refresh(self, env: MECEnvironment) -> bool:
        if env.time_step == self.last_refresh_step:
            return False
        if self.last_refresh_step < 0:
            return True
        return env.time_step % self.refresh_interval == 0

    def _should_retry_remote(self, env: MECEnvironment) -> bool:
        return (
            isinstance(self.client, RuleBasedLLMClient)
            and self.last_failure_step >= 0
            and env.time_step - self.last_failure_step >= self.retry_remote_every
        )

    def _ensure_client(self, env: MECEnvironment) -> None:
        if self._should_retry_remote(env):
            self.client = build_llm_client()

    def _refresh_policy(self, env: MECEnvironment) -> None:
        self._ensure_client(env)
        prompt = build_llm_prompt(env, self.base_params)
        try:
            response_text = self.client.generate(prompt)
            parsed = parse_llm_response(response_text, self.base_params)
            self.last_rationale = parsed.pop("rationale", "")
            self.last_error = ""
            self.active_params = replace(self.base_params, **parsed)
        except Exception as exc:
            self.active_params = self.base_params
            self.last_error = str(exc)
            self.last_rationale = f"fallback to base parameters: {exc}"
            self.client = RuleBasedLLMClient()
            self.last_failure_step = env.time_step

        self.executor = CostAwarePolicy(self.active_params)
        self.last_refresh_step = env.time_step

    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        if self._should_refresh(env):
            self._refresh_policy(env)
        return self.executor.select_node(env, user)

    def debug_snapshot(self) -> dict:
        return {
            "llm_rationale": self.last_rationale,
            "llm_provider": self.client.provider_name,
            "llm_mode": self.client.mode_name,
            "llm_error": self.last_error,
            "llm_lambda_delay": self.active_params.lambda_delay,
            "llm_lambda_migration": self.active_params.lambda_migration,
            "llm_lambda_resource": self.active_params.lambda_resource,
            "llm_lambda_balance": self.active_params.lambda_balance,
            "llm_migrate_threshold": self.active_params.migrate_threshold,
            "llm_cooldown_steps": self.active_params.cooldown_steps,
        }
