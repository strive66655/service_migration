from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict


class BaseLLMClient:
    provider_name = "unknown"
    mode_name = "remote"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class RuleBasedLLMClient(BaseLLMClient):
    """
    Local fallback for the decision layer.
    It keeps the experiment runnable when no provider key is configured or the
    remote request fails.
    """

    provider_name = "rule_based"
    mode_name = "fallback"

    def _extract_state(self, prompt: str) -> Dict[str, Any]:
        marker = "STATE_JSON:\n"
        if marker not in prompt:
            raise ValueError("Prompt missing STATE_JSON block.")
        state_text = prompt.split(marker, maxsplit=1)[1].strip()
        return json.loads(state_text)

    def generate(self, prompt: str) -> str:
        state = self._extract_state(prompt)
        base = state["base_policy_params"]

        lambda_delay = float(base["lambda_delay"])
        lambda_migration = float(base["lambda_migration"])
        lambda_resource = float(base["lambda_resource"])
        lambda_balance = float(base["lambda_balance"])
        migrate_threshold = float(base["migrate_threshold"])
        cooldown_steps = int(base["cooldown_steps"])

        avg_speed = float(state.get("avg_speed", 0.0))
        avg_latency = float(state.get("avg_latency_sensitivity", 0.0))
        max_load = float(state.get("max_node_load_ratio", 0.0))
        service_distribution = state.get("service_distribution", {})
        intent_samples = " ".join(state.get("intent_samples", [])).lower()

        ar_ratio = float(service_distribution.get("ar", 0)) / max(float(state.get("user_count", 1)), 1.0)
        realtime_pressure = 0.5 * avg_latency + 0.3 * ar_ratio + 0.2 * min(avg_speed / 5.0, 1.0)

        if realtime_pressure > 0.62:
            lambda_delay += 0.06
            lambda_migration -= 0.03
            lambda_balance -= 0.01
            migrate_threshold -= 0.02

        if max_load > 0.55:
            lambda_resource += 0.03
            lambda_balance += 0.02
            lambda_delay -= 0.03
            migrate_threshold += 0.01

        if "stable" in intent_samples or "avoid migration" in intent_samples:
            lambda_migration += 0.06
            lambda_delay -= 0.03
            cooldown_steps += 1
            migrate_threshold += 0.02

        if "low latency" in intent_samples or "real-time" in intent_samples:
            lambda_delay += 0.05
            lambda_migration -= 0.03
            cooldown_steps = max(1, cooldown_steps - 1)

        response = {
            "lambda_delay": lambda_delay,
            "lambda_migration": lambda_migration,
            "lambda_resource": lambda_resource,
            "lambda_balance": lambda_balance,
            "migrate_threshold": migrate_threshold,
            "cooldown_steps": cooldown_steps,
            "rationale": (
                "Adjusted policy weights from semantic state. "
                f"realtime_pressure={realtime_pressure:.3f}, max_load={max_load:.3f}"
            ),
        }
        return json.dumps(response, ensure_ascii=False)


class JsonHTTPClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        url: str,
        model: str,
        headers: Dict[str, str] | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.model = model
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

    def _post_json(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.headers,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM API request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc


class OpenAIResponsesClient(JsonHTTPClient):
    provider_name = "openai"
    mode_name = "remote"

    def __init__(self, api_key: str, model: str | None = None, timeout_seconds: float = 20.0) -> None:
        super().__init__(
            api_key=api_key,
            url="https://api.openai.com/v1/responses",
            model=model or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            timeout_seconds=timeout_seconds,
        )

    def _extract_output_text(self, payload: Dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text = content.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return text.strip()

        raise ValueError("No text content found in OpenAI response.")

    def generate(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "reasoning": {"effort": "low"},
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a service migration meta-controller. "
                                "Return only JSON with the requested keys and no markdown."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
        }
        payload = self._post_json(body)
        return self._extract_output_text(payload)


class OpenRouterChatClient(JsonHTTPClient):
    provider_name = "openrouter"
    mode_name = "remote"

    def __init__(self, api_key: str, model: str | None = None, timeout_seconds: float = 20.0) -> None:
        headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        title = os.getenv("OPENROUTER_APP_TITLE", "service_migration").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        super().__init__(
            api_key=api_key,
            url="https://openrouter.ai/api/v1/chat/completions",
            model=model or os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free"),
            headers=headers,
            timeout_seconds=timeout_seconds,
        )

    def _extract_output_text(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            raise ValueError("No choices found in OpenRouter response.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

        raise ValueError("No text content found in OpenRouter response.")

    def generate(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a service migration meta-controller. "
                        "Return only JSON with the requested keys and no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        payload = self._post_json(body)
        return self._extract_output_text(payload)


def build_llm_client() -> BaseLLMClient:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if api_key:
            return OpenRouterChatClient(api_key=api_key)
        return RuleBasedLLMClient()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            return OpenAIResponsesClient(api_key=api_key)
        return RuleBasedLLMClient()

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        return OpenRouterChatClient(api_key=openrouter_key)

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return OpenAIResponsesClient(api_key=openai_key)

    return RuleBasedLLMClient()
