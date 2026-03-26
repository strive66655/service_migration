from __future__ import annotations

import os

from openai import BadRequestError, OpenAI

from src.llm.base import BaseLLMProvider


class QwenProvider(BaseLLMProvider):
    _UNSUPPORTED_CHAT_MODEL_PREFIXES = ("qvq-",)

    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature

        if self.model_name.startswith(self._UNSUPPORTED_CHAT_MODEL_PREFIXES):
            raise ValueError(
                f"Qwen model '{self.model_name}' is not supported via chat.completions. "
                "Use a text chat model such as 'qwen-plus' or 'qwen-turbo', or implement the "
                "model-specific API path for this family."
            )

        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        resolved_base_url = (
            base_url
            or os.getenv("QWEN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        if not resolved_api_key:
            raise ValueError("Qwen API key is not set. Please set DASHSCOPE_API_KEY or QWEN_API_KEY.")

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except BadRequestError as exc:
            response_body = getattr(exc, "body", None) or {}
            error_message = str(response_body)
            if isinstance(response_body, dict):
                error_message = str(response_body.get("error", {}).get("message", response_body))

            if "does not support http call" in error_message:
                raise ValueError(
                    f"Qwen model '{self.model_name}' cannot be used through the current "
                    "chat.completions-compatible API for this account. Switch to a standard "
                    "text chat model such as 'qwen-plus' or 'qwen-turbo'."
                ) from exc
            raise

        return response.choices[0].message.content or ""
