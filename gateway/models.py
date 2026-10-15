from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False


class CompletionResult(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int


class ModelClient(Protocol):
    async def complete(self, model: str, messages: list[ChatMessage]) -> CompletionResult: ...

    def stream(self, model: str, messages: list[ChatMessage]) -> AsyncIterator[str]: ...
