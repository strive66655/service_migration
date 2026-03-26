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
        operator_instruction: str = "",
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_name = model_name
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser(provider_name, model_name)
        self.default_params = default_params
        self.operator_instruction = operator_instruction.strip()
        
    def suggest_params(
        self,
        env,
        current_params: Optional[PolicyParams] = None,
        recent_step_metrics: Optional[List[StepMetrics]] = None,
    ) -> LLMDecision:
        active_params = current_params or self.default_params
        scene = self._build_scene_summary(env, recent_step_metrics or [])
        system_prompt, user_prompt = self.prompt_builder.build(scene, active_params)
        try:
            raw_text = self.provider.generate(system_prompt, user_prompt)
        except Exception as exc:
            return LLMDecision(
                params=active_params.normalized(),
                reason=f"LLM provider call failed, using current parameters. Error: {exc}",
                provider=self.provider_name,
                model=self.model_name,
                used_fallback=True,
                raw_text="",
                parsed_payload={},
            )

        return self.response_parser.parse(raw_text, active_params)

    def _build_scene_summary(self, env, recent_step_metrics: List[StepMetrics]) -> SceneSummary:
        snapshot = env.snapshot()

        node_loads = [node["load_ratio"] for node in snapshot["nodes"].values()]
        avg_node_load = sum(node_loads) / len(node_loads) if node_loads else 0.0
        max_node_load = max(node_loads) if node_loads else 0.0

        service_counter = Counter(
            user_data["service_type"] for user_data in snapshot["users"].values()
        )

        # representative_intents = {}
        # for user_data in snapshot["users"].values():
        #     service_type = user_data.get("service_type", "unknown")
        #     intent_text = user_data.get("intent_text", "").strip()
        #     if intent_text and service_type not in representative_intents:
        #         representative_intents[service_type] = intent_text

        # if representative_intents:
        #     intent_summary = " ; ".join(
        #         f"{service}: {text}"
        #         for service, text in representative_intents.items()
        #     )
        # else:
        #     intent_summary = "无显式用户意图文本"
            
        if service_counter:
            user_context_summary = " ; ".join(
                f"{service}: {count} users"
                for service, count in service_counter.items()
            )
        else:
            user_context_summary = "无可用业务类型统计"

        window_metrics = recent_step_metrics
        window_steps = len(window_metrics)

        user_count = len(snapshot["users"])
        safe_user_count = max(user_count, 1)
        safe_denominator = max(window_steps * safe_user_count, 1)

        failed_allocations_recent = sum(m.failed_allocations for m in window_metrics)
        migrations_recent = sum(m.migration_count for m in window_metrics)

        avg_delay_recent = (
            sum(m.avg_delay for m in window_metrics) / window_steps
            if window_steps > 0
            else 0.0
        )

        migration_rate_recent = migrations_recent / safe_denominator
        failed_allocation_rate_recent = failed_allocations_recent / safe_denominator

        users_in_cooldown = sum(
            1
            for user_data in snapshot["users"].values()
            if int(user_data.get("cooldown_left", 0)) > 0
        )
        users_in_cooldown_ratio = users_in_cooldown / safe_user_count

        avg_migrations_per_user_recent = migrations_recent / safe_user_count

        return SceneSummary(
            step=int(snapshot["time_step"]),
            node_count=len(snapshot["nodes"]),
            user_count=len(snapshot["users"]),
            avg_node_load=float(avg_node_load),
            max_node_load=float(max_node_load),
            failed_allocations_recent=int(failed_allocations_recent),
            migrations_recent=int(migrations_recent),
            avg_delay_recent=float(avg_delay_recent),
            migration_rate_recent=float(migration_rate_recent),
            failed_allocation_rate_recent=float(failed_allocation_rate_recent),
            users_in_cooldown_ratio=float(users_in_cooldown_ratio),
            avg_migrations_per_user_recent=float(avg_migrations_per_user_recent),
            service_counts=dict(service_counter),
            user_context_summary=user_context_summary,
            operator_instruction=self.operator_instruction or "无显式运维指令，默认以网络稳定与资源效率为目标。",
            snapshot=snapshot,
        )


