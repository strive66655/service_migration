from __future__ import annotations

import os

from openai import OpenAI

from src.llm.base import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    def __init__(
        self,
        model_name: str = "openai/gpt-5.4-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature

        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"

        if not resolved_api_key:
            raise ValueError("OpenRouter API key is not set. Please set OPENROUTER_API_KEY.")

        default_headers: dict[str, str] = {}
        resolved_site_url = site_url or os.getenv("OPENROUTER_SITE_URL")
        resolved_app_name = app_name or os.getenv("OPENROUTER_APP_NAME")
        if resolved_site_url:
            default_headers["HTTP-Referer"] = resolved_site_url
        if resolved_app_name:
            default_headers["X-Title"] = resolved_app_name

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            default_headers=default_headers or None,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
