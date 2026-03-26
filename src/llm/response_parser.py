from __future__ import annotations

import json
import re
from typing import Any, Dict

from src.algorithms.policy_params import PolicyParams
from src.llm.schemas import LLMDecision


class ResponseParser:
    def __init__(self, provider_name: str, model_name: str) -> None:
        self.provider_name = provider_name
        self.model_name = model_name

    def parse(self, raw_text: str, default_params: PolicyParams) -> LLMDecision:
        try:
            payload = self._extract_json(raw_text)
            safe_partial = self._sanitize_partial(payload)
            merged = default_params.merged_with(safe_partial).normalized()

            return LLMDecision(
                params=merged,
                reason=str(payload.get("reason", "No reason provided")),
                provider=self.provider_name,
                model=self.model_name,
                used_fallback=False,
                raw_text=raw_text,
                parsed_payload=payload,
            )
        except Exception:
            return LLMDecision(
                params=default_params.normalized(),
                reason="Failed to parse LLM response, using default parameters.",
                provider=self.provider_name,
                model=self.model_name,
                used_fallback=True,
                raw_text=raw_text,
                parsed_payload={},
            )

    
    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the response.")
        
        candidate = match.group(0)
        data = json.loads(candidate)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not an object.")
        return data

    def _sanitize_partial(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}

        float_fields = [
            "lambda_delay",
            "lambda_migration",
            "lambda_resource",
            "lambda_balance",
            "migrate_threshold",
        ]
        int_fields = ["cooldown_steps"]

        for key in float_fields:
            if key in payload:
                safe[key] = float(payload[key])
            
        for key in int_fields:
            if key in payload:
                safe[key] = int(payload[key])

        return safe
                
    
    