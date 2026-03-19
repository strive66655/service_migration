from __future__ import annotations

import json
from typing import Any, Dict

from src.llm_core.mode_selector import MODE_DESCRIPTIONS


def build_llm_prompt(summary: Dict[str, Any]) -> str:
    state_json = json.dumps(summary, ensure_ascii=False, indent=2)
    mode_lines = "\n".join(
        f"- {name}: {description}" for name, description in MODE_DESCRIPTIONS.items()
    )
    return (
        "You are a high-level MEC service migration policy selector.\n"
        "Choose exactly one mode from the fixed set below.\n"
        f"{mode_lines}\n"
        "Use the current summary to pick the single best mode.\n"
        "Return JSON only with keys: mode, reason.\n"
        "STATE_JSON:\n"
        f"{state_json}"
    )
