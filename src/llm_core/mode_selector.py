from __future__ import annotations

from typing import Any, Dict, Tuple


MODE_TEMPLATES: Dict[str, Dict[str, float | int]] = {
    "latency_first": {
        "lambda_delay": 0.50,
        "lambda_migration": 0.15,
        "lambda_resource": 0.20,
        "lambda_balance": 0.15,
        "migrate_threshold": 0.06,
        "cooldown_steps": 3,
    },
    "stability_first": {
        "lambda_delay": 0.30,
        "lambda_migration": 0.35,
        "lambda_resource": 0.20,
        "lambda_balance": 0.15,
        "migrate_threshold": 0.15,
        "cooldown_steps": 6,
    },
    "avoid_migration": {
        "lambda_delay": 0.25,
        "lambda_migration": 0.40,
        "lambda_resource": 0.20,
        "lambda_balance": 0.15,
        "migrate_threshold": 0.18,
        "cooldown_steps": 8,
    },
    "resource_relief": {
        "lambda_delay": 0.25,
        "lambda_migration": 0.20,
        "lambda_resource": 0.35,
        "lambda_balance": 0.20,
        "migrate_threshold": 0.10,
        "cooldown_steps": 4,
    },
}

MODE_DESCRIPTIONS: Dict[str, str] = {
    "latency_first": "prioritize low latency and follow user mobility",
    "stability_first": "reduce unnecessary migrations and keep placements stable",
    "avoid_migration": "suppress oscillation when recent migrations are frequent or benefit is small",
    "resource_relief": "relieve overloaded nodes and reduce failed allocations",
}

DEFAULT_MODE = "stability_first"


def normalize_mode(mode: Any) -> str:
    candidate = str(mode or "").strip().lower()
    if candidate in MODE_TEMPLATES:
        return candidate
    return DEFAULT_MODE


def fallback_select_mode(summary: Dict[str, Any]) -> Tuple[str, str]:
    if (
        int(summary.get("recent_failed_allocations_last_3", 0)) > 0
        or float(summary.get("max_load_ratio", 0.0)) > 0.8
    ):
        return "resource_relief", "Fallback: overload or failed allocations detected."

    if bool(summary.get("boundary_risk")) or int(summary.get("recent_migrations_last_3", 0)) >= 2:
        return "avoid_migration", "Fallback: oscillation risk detected."

    if (
        float(summary.get("latency_sensitive_ratio", 0.0)) > 0.5
        and float(summary.get("avg_user_speed", 0.0)) > 2.5
    ):
        return "latency_first", "Fallback: latency-sensitive mobile services dominate."

    return DEFAULT_MODE, "Fallback: stable default mode."
