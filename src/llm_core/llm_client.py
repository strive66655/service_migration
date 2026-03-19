from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

from src.llm_core.mode_selector import fallback_select_mode


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
        mode, reason = fallback_select_mode(state)
        response = {
            "mode": mode,
            "reason": reason,
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


class OpenAICompatibleChatClient(JsonHTTPClient):
    provider_name = "openai_compatible"
    mode_name = "local"

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "local",
        headers: Dict[str, str] | None = None,
        timeout_seconds: float = 20.0,
        provider_name: str = "openai_compatible",
        mode_name: str = "local",
    ) -> None:
        super().__init__(
            api_key=api_key,
            url=url,
            model=model,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
        self.provider_name = provider_name
        self.mode_name = mode_name

    def _extract_output_text(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            raise ValueError(f"No choices found in {self.provider_name} response.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts)

        raise ValueError(f"No text content found in {self.provider_name} response.")

    def generate(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a service migration meta-controller. "
                        "Return only JSON with keys mode and reason, with no markdown."
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
                                "Return only JSON with keys mode and reason, with no markdown."
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
                        "Return only JSON with keys mode and reason, with no markdown."
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


class OllamaChatClient(OpenAICompatibleChatClient):
    provider_name = "ollama"
    mode_name = "local"

    def __init__(self, model: str | None = None, timeout_seconds: float = 20.0) -> None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        super().__init__(
            url=f"{base_url}/v1/chat/completions",
            model=model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama").strip() or "ollama",
            timeout_seconds=timeout_seconds,
            provider_name=self.provider_name,
            mode_name=self.mode_name,
        )


class LocalOpenAICompatibleClient(OpenAICompatibleChatClient):
    provider_name = "local"
    mode_name = "local"

    def __init__(self, timeout_seconds: float = 20.0) -> None:
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1").strip().rstrip("/")
        model = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-7b-instruct")
        api_key = os.getenv("LOCAL_LLM_API_KEY", "local").strip() or "local"
        super().__init__(
            url=f"{base_url}/chat/completions",
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            provider_name=self.provider_name,
            mode_name=self.mode_name,
        )


def build_llm_client() -> BaseLLMClient:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider == "local":
        return LocalOpenAICompatibleClient()

    if provider == "ollama":
        return OllamaChatClient()

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

    local_base_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip()
    if local_base_url:
        return LocalOpenAICompatibleClient()

    ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
    if ollama_model:
        return OllamaChatClient()

    return RuleBasedLLMClient()
