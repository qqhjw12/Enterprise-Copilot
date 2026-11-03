"""可观测性接线测试：trace 透传、span 语义、日志字段。"""
from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from common.middleware import RequestContextMiddleware
from runtime.obs.spans import gen_ai_attrs, parse_traceparent, span


def test_parse_traceparent_accepts_w3c_and_rejects_garbage():
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"

    assert parse_traceparent(f"00-{trace_id}-00f067aa0ba902b7-01") == trace_id
    assert parse_traceparent("00-not-a-trace-id-01") is None
    assert parse_traceparent(None) is None


@pytest.mark.asyncio
async def test_middleware_propagates_incoming_trace_id():
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware, service="test")

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/ping",
            headers={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
        )

    assert response.headers["x-trace-id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert response.headers["x-request-id"].startswith("req_")


def test_span_uses_genai_semconv(caplog):
    caplog.set_level(logging.INFO, logger="trace")
    attrs = gen_ai_attrs(operation="chat", model="deepseek-chat", tool="get_employee_assets")

    with span("chat", **attrs) as current:
        current.set(**{"gen_ai.usage.input_tokens": 12, "gen_ai.usage.output_tokens": 34})

    records = [r for r in caplog.records if r.name == "trace"]
    assert records, "span 必须落结构化日志"
    payload = records[-1].span
    assert payload["name"] == "chat"
    assert len(payload["trace_id"]) == 32
    assert payload["attributes"]["gen_ai.operation.name"] == "chat"
    assert payload["attributes"]["gen_ai.request.model"] == "deepseek-chat"
    assert payload["attributes"]["gen_ai.tool.name"] == "get_employee_assets"
    assert payload["attributes"]["gen_ai.usage.output_tokens"] == 34
    assert payload["status"] == "ok"
