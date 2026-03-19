from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import replace
from typing import Any, Deque, Dict, Optional

from src.algorithms.base_policy import BasePolicy
from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import User
from src.env.mec_env import MECEnvironment
from src.llm_core.llm_client import BaseLLMClient, RuleBasedLLMClient, build_llm_client
from src.llm_core.mode_selector import DEFAULT_MODE, MODE_TEMPLATES, fallback_select_mode
from src.llm_core.parser import parse_llm_response
from src.llm_core.prompt_builder import build_llm_prompt
from src.utils.metrics import StepMetrics


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
        self.last_mode = DEFAULT_MODE
        self.last_provider = "unknown"
        self.last_trigger_reason = "init"
        self.last_mode_changed = False
        self.last_action_changed_by_llm = False
        self.current_step = -1
        self.current_summary: Dict[str, Any] = {}
        self.client: BaseLLMClient = build_llm_client()
        self.executor = CostAwarePolicy(self.active_params)
        self.base_executor = CostAwarePolicy(self.base_params)
        self.recent_migrations: Deque[int] = deque(maxlen=5)
        self.recent_failed_allocations: Deque[int] = deque(maxlen=5)
        self.nearest_node_history: Dict[int, Deque[int]] = {}
        self.last_dominant_intent = ""
        self.last_observed_load_stats = {
            "avg_load_ratio": 0.0,
            "max_load_ratio": 0.0,
            "min_load_ratio": 0.0,
            "load_std": 0.0,
        }

    def _should_retry_remote(self, env: MECEnvironment) -> bool:
        return (
            isinstance(self.client, RuleBasedLLMClient)
            and self.last_failure_step >= 0
            and env.time_step - self.last_failure_step >= self.retry_remote_every
        )

    def _ensure_client(self, env: MECEnvironment) -> None:
        if self._should_retry_remote(env):
            self.client = build_llm_client()

    def _categorize_intent(self, user: User) -> str:
        text = user.intent_text.lower()
        service_name = user.service_type.value

        if service_name in {"ar", "video"}:
            return "latency_sensitive"
        if service_name == "background":
            return "stability_sensitive"

        latency_keywords = ("low latency", "real-time", "realtime", "ar", "video", "render")
        stability_keywords = ("stable", "stability", "avoid migration", "medical", "industrial")
        throughput_keywords = ("batch", "throughput", "sync", "compute", "background")

        if any(keyword in text for keyword in latency_keywords):
            return "latency_sensitive"
        if any(keyword in text for keyword in stability_keywords):
            return "stability_sensitive"
        if any(keyword in text for keyword in throughput_keywords):
            return "throughput_sensitive"
        return "throughput_sensitive"

    def _update_nearest_history(self, env: MECEnvironment) -> None:
        for user in env.users.values():
            if not env.nodes:
                continue
            nearest_node_id = min(
                env.nodes.values(),
                key=lambda node: env.distance(user, node),
            ).node_id
            history = self.nearest_node_history.setdefault(user.user_id, deque(maxlen=3))
            history.append(nearest_node_id)

    def _compute_boundary_risk(self, env: MECEnvironment) -> bool:
        for user in env.users.values():
            distances = sorted((env.distance(user, node), node.node_id) for node in env.nodes.values())
            if len(distances) < 2:
                continue

            nearest_gap = distances[1][0] - distances[0][0]
            history = self.nearest_node_history.get(user.user_id, deque())
            changes = sum(1 for idx in range(1, len(history)) if history[idx] != history[idx - 1])

            if nearest_gap <= max(3.0, distances[0][0] * 0.15) and changes >= 2:
                return True

        return False

    def _build_summary(self, env: MECEnvironment) -> Dict[str, Any]:
        speeds = [((user.vx ** 2 + user.vy ** 2) ** 0.5) for user in env.users.values()]
        latency_sensitivities = [user.workload.latency_sensitivity for user in env.users.values()]
        intent_counter = Counter(self._categorize_intent(user) for user in env.users.values())

        dominant_intent = ""
        if intent_counter:
            dominant_intent = max(intent_counter, key=intent_counter.get)

        latency_count = intent_counter.get("latency_sensitive", 0)
        user_count = max(len(env.users), 1)
        boundary_risk = self._compute_boundary_risk(env)
        dominant_intent_changed = bool(self.last_dominant_intent) and dominant_intent != self.last_dominant_intent

        summary = {
            "time_step": env.time_step,
            "user_count": len(env.users),
            "node_count": len(env.nodes),
            "avg_user_speed": round(statistics.mean(speeds), 4) if speeds else 0.0,
            "avg_latency_sensitivity": round(statistics.mean(latency_sensitivities), 4)
            if latency_sensitivities
            else 0.0,
            "recent_migrations_last_3": int(sum(list(self.recent_migrations)[-3:])),
            "recent_migrations_last_5": int(sum(self.recent_migrations)),
            "recent_failed_allocations_last_3": int(sum(list(self.recent_failed_allocations)[-3:])),
            "recent_failed_allocations_last_5": int(sum(self.recent_failed_allocations)),
            "avg_load_ratio": round(float(self.last_observed_load_stats["avg_load_ratio"]), 4),
            "max_load_ratio": round(float(self.last_observed_load_stats["max_load_ratio"]), 4),
            "min_load_ratio": round(float(self.last_observed_load_stats["min_load_ratio"]), 4),
            "load_std": round(float(self.last_observed_load_stats["load_std"]), 4),
            "load_imbalance": round(
                float(self.last_observed_load_stats["max_load_ratio"])
                - float(self.last_observed_load_stats["min_load_ratio"]),
                4,
            ),
            "boundary_risk": boundary_risk,
            "dominant_intents": dict(intent_counter),
            "dominant_intent": dominant_intent,
            "dominant_intent_changed": dominant_intent_changed,
            "latency_sensitive_ratio": round(latency_count / user_count, 4),
            "base_policy_params": {
                "lambda_delay": self.base_params.lambda_delay,
                "lambda_migration": self.base_params.lambda_migration,
                "lambda_resource": self.base_params.lambda_resource,
                "lambda_balance": self.base_params.lambda_balance,
                "migrate_threshold": self.base_params.migrate_threshold,
                "cooldown_steps": self.base_params.cooldown_steps,
            },
        }
        self.last_dominant_intent = dominant_intent
        return summary

    def _prepare_step(self, env: MECEnvironment) -> None:
        if env.time_step == self.current_step:
            return

        self.current_step = env.time_step
        self.last_mode_changed = False
        self.last_action_changed_by_llm = False
        self.last_trigger_reason = "reuse_previous"
        self._update_nearest_history(env)
        self.current_summary = self._build_summary(env)

    def _refresh_reason(self, env: MECEnvironment) -> str | None:
        if env.time_step == self.last_refresh_step:
            return None
        if self.last_refresh_step < 0:
            return "step_init"
        if int(self.current_summary.get("recent_failed_allocations_last_3", 0)) > 0:
            return "failed_allocations"
        if int(self.current_summary.get("recent_migrations_last_3", 0)) >= 2:
            return "recent_churn"
        if float(self.current_summary.get("max_load_ratio", 0.0)) > 0.75:
            return "load_spike"
        if bool(self.current_summary.get("boundary_risk")):
            return "boundary_risk"
        if bool(self.current_summary.get("dominant_intent_changed")):
            return "dominant_intent_changed"
        if self.refresh_interval > 1 and env.time_step - self.last_refresh_step >= self.refresh_interval:
            return "periodic_refresh"
        return None

    def _apply_mode(self, mode: str) -> None:
        template = MODE_TEMPLATES[mode]
        self.active_params = replace(self.base_params, **template)
        self.executor = CostAwarePolicy(self.active_params)

    def _refresh_policy(self, env: MECEnvironment, trigger_reason: str) -> None:
        self._ensure_client(env)
        prompt = build_llm_prompt(self.current_summary)
        previous_mode = self.last_mode
        provider_name = self.client.provider_name

        try:
            response_text = self.client.generate(prompt)
            parsed = parse_llm_response(response_text)
            mode = parsed["mode"]
            reason = parsed.get("reason", "")
            self.last_error = ""
        except Exception as exc:
            mode, reason = fallback_select_mode(self.current_summary)
            self.last_error = str(exc)
            reason = f"{reason} Remote error: {exc}"
            self.client = RuleBasedLLMClient()
            self.last_failure_step = env.time_step
            provider_name = self.client.provider_name

        self.last_mode = mode
        self.last_provider = provider_name
        self.last_rationale = reason
        self.last_trigger_reason = trigger_reason
        self.last_mode_changed = mode != previous_mode
        self._apply_mode(mode)
        self.last_refresh_step = env.time_step

    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        self._prepare_step(env)
        trigger_reason = self._refresh_reason(env)
        if trigger_reason is not None:
            self._refresh_policy(env, trigger_reason)

        chosen_node = self.executor.select_node(env, user)
        base_node = self.base_executor.select_node(env, user)
        if chosen_node != base_node:
            self.last_action_changed_by_llm = True
        return chosen_node

    def observe_step_result(self, step_metrics: StepMetrics) -> None:
        self.recent_migrations.append(int(step_metrics.migration_count))
        self.recent_failed_allocations.append(int(step_metrics.failed_allocations))
        self.last_observed_load_stats = {
            "avg_load_ratio": float(step_metrics.avg_load_ratio),
            "max_load_ratio": float(step_metrics.max_load_ratio),
            "min_load_ratio": float(step_metrics.min_load_ratio),
            "load_std": float(step_metrics.load_std),
        }

    def debug_snapshot(self) -> dict:
        return {
            "llm_rationale": self.last_rationale,
            "llm_provider": self.last_provider,
            "llm_mode": self.last_mode,
            "llm_error": self.last_error,
            "llm_lambda_delay": self.active_params.lambda_delay,
            "llm_lambda_migration": self.active_params.lambda_migration,
            "llm_lambda_resource": self.active_params.lambda_resource,
            "llm_lambda_balance": self.active_params.lambda_balance,
            "llm_migrate_threshold": self.active_params.migrate_threshold,
            "llm_cooldown_steps": self.active_params.cooldown_steps,
            "llm_mode_changed": self.last_mode_changed,
            "llm_trigger_reason": self.last_trigger_reason,
            "action_changed_by_llm": self.last_action_changed_by_llm,
        }
