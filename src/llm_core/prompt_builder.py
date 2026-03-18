from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict

from src.algorithms.policy_params import PolicyParams
from src.env.mec_env import MECEnvironment


def build_llm_context(env: MECEnvironment, params: PolicyParams) -> Dict[str, Any]:
    service_counter = Counter(user.service_type.value for user in env.users.values())
    intent_texts = [user.intent_text.strip() for user in env.users.values() if user.intent_text.strip()]

    avg_speed = 0.0
    avg_latency_sensitivity = 0.0
    if env.users:
        avg_speed = sum((user.vx ** 2 + user.vy ** 2) ** 0.5 for user in env.users.values()) / len(env.users)
        avg_latency_sensitivity = sum(
            user.workload.latency_sensitivity for user in env.users.values()
        ) / len(env.users)

    node_loads = {str(node.node_id): round(node.load_ratio, 4) for node in env.nodes.values()}
    max_load_ratio = max((node.load_ratio for node in env.nodes.values()), default=0.0)
    avg_load_ratio = sum(node.load_ratio for node in env.nodes.values()) / max(len(env.nodes), 1)

    return {
        "time_step": env.time_step,
        "user_count": len(env.users),
        "node_count": len(env.nodes),
        "avg_speed": round(avg_speed, 4),
        "avg_latency_sensitivity": round(avg_latency_sensitivity, 4),
        "avg_node_load_ratio": round(avg_load_ratio, 4),
        "max_node_load_ratio": round(max_load_ratio, 4),
        "service_distribution": dict(service_counter),
        "intent_samples": intent_texts[:5],
        "node_loads": node_loads,
        "base_policy_params": {
            "lambda_delay": params.lambda_delay,
            "lambda_migration": params.lambda_migration,
            "lambda_resource": params.lambda_resource,
            "lambda_balance": params.lambda_balance,
            "migrate_threshold": params.migrate_threshold,
            "cooldown_steps": params.cooldown_steps,
        },
    }


def build_llm_prompt(env: MECEnvironment, params: PolicyParams) -> str:
    context = build_llm_context(env, params)
    state_json = json.dumps(context, ensure_ascii=False, indent=2)
    return (
        "You are the decision layer of a mobile edge service migration system.\n"
        "Read the environment state and adjust the low-level migration policy.\n"
        "Return valid JSON with keys: lambda_delay, lambda_migration, lambda_resource, "
        "lambda_balance, migrate_threshold, cooldown_steps, rationale.\n"
        "The four lambda values must sum to 1.0.\n"
        "Prefer lower delay for latency-sensitive or fast-moving users, and prefer higher "
        "migration penalty when stability is more important.\n"
        "STATE_JSON:\n"
        f"{state_json}"
    )
