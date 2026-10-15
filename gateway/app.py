import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.database import MetricRepository
from gateway.models import ChatCompletionRequest, CompletionResult, ModelClient
from gateway.provider import OpenAICompatibleClient, ProviderRouter
from gateway.settings import GatewaySettings


def create_app(
    database_url: str | None = None,
    model_client: ModelClient | None = None,
) -> FastAPI:
    settings = GatewaySettings()
    repository = MetricRepository(database_url or settings.database_url)
    client = model_client or ProviderRouter(
        {
            "openai": OpenAICompatibleClient(settings.openai_base_url, settings.openai_api_key),
            "ollama": OpenAICompatibleClient(settings.ollama_base_url, ""),
        }
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await repository.create_tables()
        yield
        await repository.close()

    app = FastAPI(title="LLM Gateway Lab", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest):
        started_at = time.perf_counter()
        if payload.stream:
            return StreamingResponse(
                stream_completion(payload, client, repository, started_at),
                media_type="text/event-stream",
            )
        result = await client.complete(payload.model, payload.messages)
        await save_metric(repository, payload.model, started_at, result)
        return JSONResponse(openai_completion(payload.model, result))

    @app.get("/internal/request-metrics")
    async def request_metrics():
        metrics = await repository.list_recent()
        return {"items": [metric_item(metric) for metric in metrics]}

    return app


async def stream_completion(
    payload: ChatCompletionRequest,
    client: ModelClient,
    repository: MetricRepository,
    started_at: float,
) -> AsyncIterator[str]:
    content_parts: list[str] = []
    async for content in client.stream(payload.model, payload.messages):
        content_parts.append(content)
        yield f"data: {json.dumps(stream_chunk(payload.model, content), ensure_ascii=False)}\n\n"
    result = CompletionResult(
        content="".join(content_parts),
        prompt_tokens=0,
        completion_tokens=len(content_parts),
    )
    await save_metric(repository, payload.model, started_at, result)
    yield "data: [DONE]\n\n"


async def save_metric(repository: MetricRepository, model: str, started_at: float, result: CompletionResult) -> None:
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    await repository.save(model, latency_ms, result.prompt_tokens, result.completion_tokens)


def openai_completion(model: str, result: CompletionResult) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def stream_chunk(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


def metric_item(metric) -> dict:
    return {
        "model": metric.model,
        "latency_ms": metric.latency_ms,
        "prompt_tokens": metric.prompt_tokens,
        "completion_tokens": metric.completion_tokens,
        "estimated_cost_microusd": metric.estimated_cost_microusd,
    }


app = create_app()
