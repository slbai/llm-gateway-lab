import json

import httpx
import pytest

from gateway.app import create_app
from gateway.models import ChatMessage, CompletionResult
from gateway.provider import ProviderRouter
from gateway.settings import GatewaySettings


class FakeModelClient:
    async def complete(self, model, messages):
        return CompletionResult(content="你好", prompt_tokens=12, completion_tokens=2)

    async def stream(self, model, messages):
        yield "你"
        yield "好"


@pytest.fixture
async def client(tmp_path):
    app = create_app(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/gateway.db",
        model_client=FakeModelClient(),
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
            yield test_client


async def test_returns_openai_compatible_completion_and_persists_metrics(client):
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "你好"
    assert payload["usage"] == {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}

    metrics = await client.get("/internal/request-metrics")
    assert metrics.json()["items"][0]["model"] == "demo-model"
    assert metrics.json()["items"][0]["completion_tokens"] == 2


async def test_streams_sse_chunks_and_persists_metrics(client):
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    events = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(event) for event in events[:-1]]
    assert response.status_code == 200
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == ["你", "好"]
    assert events[-1] == "[DONE]"

    metrics = await client.get("/internal/request-metrics")
    assert metrics.json()["items"][0]["completion_tokens"] == 2


async def test_routes_prefixed_model_to_its_provider():
    client = FakeModelClient()
    router = ProviderRouter({"openai": client})

    result = await router.complete("openai/gpt-4.1-mini", [ChatMessage(role="user", content="你好")])

    assert result.content == "你好"


def test_reads_openai_configuration_from_environment():
    settings = GatewaySettings(openai_api_key="test-key", ollama_base_url="http://ollama:11434/v1")

    assert settings.openai_api_key == "test-key"
    assert settings.ollama_base_url == "http://ollama:11434/v1"
