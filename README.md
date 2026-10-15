# LLM Gateway Lab

一个用于学习生产级大模型服务的 FastAPI 流式网关。第一周聚焦模型调用、流式输出与请求指标，后续再增加限流、评测和可观测性。

## 当前能力

- 提供 OpenAI 兼容的 `POST /v1/chat/completions` 接口。
- 使用 Server-Sent Events 返回流式响应。
- 按 `provider/model` 路由模型：`openai/gpt-4.1-mini` 与 `ollama/qwen2.5:7b`。
- 将模型名、耗时、输入 Token、输出 Token 和预估成本持久化到 PostgreSQL。
- 提供仅用于本地开发的 `GET /internal/request-metrics` 指标查询接口。

## 本地运行

需要 Python 3.11 以上与 PostgreSQL。复制环境变量示例并填写 OpenAI Key：

```bash
cp .env.example .env
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn gateway.app:app --reload
```

请求 OpenAI：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-4.1-mini","messages":[{"role":"user","content":"用一句话介绍杭州"}],"stream":true}'
```

如本机已运行 Ollama，可将模型改为 `ollama/<本地模型名>`。Ollama 的 OpenAI 兼容服务默认地址为 `http://localhost:11434/v1`。

## 测试

```bash
.venv/bin/python -m pytest -q
```

## 容器运行

已提供 PostgreSQL 与网关的 Compose 配置：

```bash
OPENAI_API_KEY=你的密钥 docker compose up --build
```

本机未安装 Docker 时，可按“本地运行”方式启动服务。
