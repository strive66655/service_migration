from __future__ import annotations

from collections import Counter
from typing import List, Optional

from src.algorithms.policy_params import PolicyParams
from src.llm.base import BaseLLMProvider
from src.llm.prompt_builder import PromptBuilder
from src.llm.schemas import LLMDecision, SceneSummary
from src.llm.response_parser import ResponseParser
from src.utils.metrics import StepMetrics


class LLMPolicyController:
    def __init__(
        self,
        provider: BaseLLMProvider,
        provider_name: str,
        model_name: str,
        default_params: PolicyParams,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_name = model_name
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser(provider_name, model_name)
        self.default_params = default_params
        
    def suggest_params(
        self,
        env,
        recent_step_metrics: Optional[List[StepMetrics]] = None,
    ) -> LLMDecision:
        scene = self._build_scene_summary(env, recent_step_metrics or [])
        system_prompt, user_prompt = self.prompt_builder.build(scene, self.default_params)
        raw_text =self.provider.generate(system_prompt, user_prompt)

        return self.response_parser.parse(raw_text, self.default_params)

    def _build_scene_summary(self, env, recent_step_metrics: List[StepMetrics]) -> SceneSummary:
        snapshot = env.snapshot()

        node_loads = [node["load_ratio"] for node in snapshot["nodes"].values()]
        avg_node_load = sum(node_loads) / len(node_loads) if node_loads else 0.0
        max_node_load = max(node_loads) if node_loads else 0.0

        service_counter = Counter(
            user_data["service_type"] for user_data in snapshot["users"].values()
        )

        intent_texts = [
            user_data["intent_text"].strip()
            for user_data in snapshot["users"].values()
            if user_data["intent_text"].strip()
        ]

        # HACK: 只是简单的把用户的 intent_text 拼起来，实际可以更智能地总结提炼
        intent_summary = " | ".join(intent_texts[:5])

        failed_allocations_recent = sum(m.failed_allocations for m in recent_step_metrics[-3:])
        migrations_recent = sum(m.migration_count for m in recent_step_metrics[-3:])

        return SceneSummary(
            step=int(snapshot["time_step"]),
            node_count=len(snapshot["nodes"]),
            user_count=len(snapshot["users"]),
            avg_node_load=float(avg_node_load),
            max_node_load=float(max_node_load),
            failed_allocations_recent=int(failed_allocations_recent),
            migrations_recent=int(migrations_recent),
            service_counts=dict(service_counter),
            intent_summary=intent_summary,
            snapshot=snapshot,
        )


