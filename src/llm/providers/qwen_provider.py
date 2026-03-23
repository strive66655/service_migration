from __future__ import annotations

import os
from openai import OpenAI

from src.llm.base import BaseLLMProvider


class QwenProvider(BaseLLMProvider):
    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct", # 可以替换为其他Qwen模型，如"qwen2.5-14b-instruct"
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature

        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        resolved_base_url = base_url or os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # resolved_base_url = base_url or os.getenv("QWEN_BASE_URL") or "https://openrouter.ai/api/v1"


        if not resolved_api_key:
            raise ValueError("Qwen API key is not set. Please set DASHSCOPE_API_KEY or QWEN_API_KEY.")

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
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

        