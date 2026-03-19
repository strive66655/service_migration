from __future__ import annotations

import json
import re
from typing import Any, Dict

from src.llm_core.mode_selector import normalize_mode


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


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    data = _load_response_json(response_text)
    return {
        "mode": normalize_mode(data.get("mode")),
        "reason": str(data.get("reason", "")).strip(),
    }
