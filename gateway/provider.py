import json
from collections.abc import AsyncIterator

import httpx

from gateway.models import ChatMessage, CompletionResult


class ProviderRouter:
    def __init__(self, providers: dict[str, object]):
        self.providers = providers

    def resolve(self, model: str) -> tuple[object, str]:
        provider_name, separator, provider_model = model.partition("/")
        if not separator or provider_name not in self.providers:
            raise ValueError("模型名称必须使用 provider/model 格式，且提供方必须已配置")
        return self.providers[provider_name], provider_model

    async def complete(self, model: str, messages: list[ChatMessage]) -> CompletionResult:
        provider, provider_model = self.resolve(model)
        return await provider.complete(provider_model, messages)

    async def stream(self, model: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        provider, provider_model = self.resolve(model)
        async for content in provider.stream(provider_model, messages):
            yield content


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def complete(self, model: str, messages: list[ChatMessage]) -> CompletionResult:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "messages": [message.model_dump() for message in messages]},
            )
            response.raise_for_status()
            payload = response.json()
        usage = payload.get("usage", {})
        return CompletionResult(
            content=payload["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def stream(self, model: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "messages": [message.model_dump() for message in messages], "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = line.removeprefix("data: ")
                    if event == "[DONE]":
                        return
                    content = json.loads(event)["choices"][0].get("delta", {}).get("content")
                    if content:
                        yield content
