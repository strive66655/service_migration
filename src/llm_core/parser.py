from __future__ import annotations

import json
import re
from typing import Any, Dict

from src.algorithms.policy_params import PolicyParams


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _extract_fenced_json(response_text: str) -> str | None:
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_first_json_object(response_text: str) -> str | None:
    start = response_text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(response_text)):
        ch = response_text[idx]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return response_text[start : idx + 1].strip()

    return None


def _load_response_json(response_text: str) -> Dict[str, Any]:
    candidates = [
        response_text.strip(),
        _extract_fenced_json(response_text),
        _extract_first_json_object(response_text),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    raise ValueError("Could not extract a valid JSON object from LLM response.")


def parse_llm_response(response_text: str, base_params: PolicyParams) -> Dict[str, Any]:
    data = _load_response_json(response_text)

    weights = {
        "lambda_delay": float(data.get("lambda_delay", base_params.lambda_delay)),
        "lambda_migration": float(data.get("lambda_migration", base_params.lambda_migration)),
        "lambda_resource": float(data.get("lambda_resource", base_params.lambda_resource)),
        "lambda_balance": float(data.get("lambda_balance", base_params.lambda_balance)),
    }

    for key, value in weights.items():
        weights[key] = _clamp(value, 0.05, 0.8)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Invalid weight sum from LLM response.")

    for key in weights:
        weights[key] = weights[key] / total_weight

    migrate_threshold = float(data.get("migrate_threshold", base_params.migrate_threshold))
    cooldown_steps = int(data.get("cooldown_steps", base_params.cooldown_steps))

    parsed = {
        **weights,
        "migrate_threshold": _clamp(migrate_threshold, 0.02, 0.25),
        "cooldown_steps": int(_clamp(cooldown_steps, 1, 6)),
        "rationale": str(data.get("rationale", "")).strip(),
    }
    return parsed
