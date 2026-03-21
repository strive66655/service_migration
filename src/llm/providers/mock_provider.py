from __future__ import annotations

import json

from src.llm.base import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = {
            "lambda_delay": 0.42,
            "lambda_migration": 0.18,
            "lambda_resource": 0.24,
            "lambda_balance": 0.16,
            "migrate_threshold": 0.08,
            "cooldown_steps": 4,
            "reason": "mock_response_for_local_debug",
        }
        return json.dumps(response, ensure_ascii=False)