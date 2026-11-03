"""UI Message Stream 协议测试：按 AI SDK v1 的 part 序列断言，不做模糊匹配。"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from runtime.api import chat
from runtime.api.deps import get_caller
from runtime.api.ui_stream import STREAM_HEADER, STREAM_VERSION, to_ui_message_stream
from runtime.mcp_layer.schemas import CallerContext


async def _events(items: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for item in items:
        yield item


def _parse(chunks: list[str]) -> tuple[list[dict[str, Any]], bool]:
    parts: list[dict[str, Any]] = []
    done = False
    for chunk in chunks:
        for line in chunk.strip().splitlines():
            assert line.startswith("data: "), line
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                continue
            parts.append(json.loads(payload))
    return parts, done


@pytest.mark.asyncio
async def test_text_flow_maps_to_start_delta_end():
    events = [
        {"event": "start", "data": {"thread_id": "t1"}},
        {"event": "route", "data": {"intent": "knowledge"}},
        {"event": "token", "data": {"delta": "年假"}},
        {"event": "token", "data": {"delta": "10 天"}},
        {"event": "citations", "data": {"citations": [{"title": "员工年假与请假制度"}]}},
        {"event": "done", "data": {"thread_id": "t1", "status": "ok", "citations": [], "usage": {}}},
    ]

    chunks = [chunk async for chunk in to_ui_message_stream(_events(events), message_id="msg_1")]
    parts, done = _parse(chunks)
    kinds = [p["type"] for p in parts]

    assert done is True
    assert kinds == [
        "start",
        "start-step",
        "data-route",
        "text-start",
        "text-delta",
        "text-delta",
        "data-citations",
        "text-end",
        "finish-step",
        "data-done",
        "finish",
    ]
    assert "".join(p.get("delta", "") for p in parts) == "年假10 天"


@pytest.mark.asyncio
async def test_tool_calls_and_approval_keep_ids_consistent():
    events = [
        {"event": "start", "data": {}},
        {"event": "tool_call", "data": {"calls": [{"tool": "get_employee_assets", "args": {"employee_id": "E1024"}, "requires_approval": False}]}},
        {"event": "tool_result", "data": {"tool": "get_employee_assets", "ok": True, "data": {"assets": 2}}},
        {"event": "tool_call", "data": {"calls": [{"tool": "submit_leave_request", "args": {"employee_id": "E1024"}, "requires_approval": True}]}},
        {"event": "interrupt", "data": {"thread_id": "t2", "approval": {"calls": []}, "budget": {}}},
        {"event": "done", "data": {"thread_id": "t2", "status": "awaiting_approval", "awaiting_approval": True}},
    ]

    chunks = [chunk async for chunk in to_ui_message_stream(_events(events), message_id="msg_2")]
    parts, _ = _parse(chunks)

    inputs = [p for p in parts if p["type"] == "tool-input-available"]
    outputs = [p for p in parts if p["type"] == "tool-output-available"]
    approvals = [p for p in parts if p["type"] == "tool-approval-request"]

    assert [p["toolName"] for p in inputs] == ["get_employee_assets", "submit_leave_request"]
    assert outputs[0]["toolCallId"] == inputs[0]["toolCallId"]
    assert outputs[0]["output"]["ok"] is True
    assert approvals[0]["toolCallId"] == inputs[1]["toolCallId"]
    assert any(p["type"] == "data-approval" for p in parts)


@pytest.mark.asyncio
async def test_error_part_terminates_stream():
    events = [{"event": "start", "data": {}}, {"event": "error", "data": {"message": "执行失败：boom"}}]

    chunks = [chunk async for chunk in to_ui_message_stream(_events(events), message_id="msg_3")]
    parts, done = _parse(chunks)

    assert done is True
    assert parts[-1]["type"] == "finish"
    assert parts[-2]["type"] == "finish-step"
    assert parts[-3]["type"] == "error"
    assert "boom" in parts[-3]["errorText"]


@pytest.mark.asyncio
async def test_blocked_done_is_rendered_as_text():
    events = [
        {"event": "start", "data": {}},
        {"event": "guardrail", "data": {"stage": "input", "action": "block", "rules": ["override_instructions"]}},
        {"event": "done", "data": {"status": "blocked", "answer": "请求已被安全策略拦截。"}},
    ]

    chunks = [chunk async for chunk in to_ui_message_stream(_events(events), message_id="msg_4")]
    parts, done = _parse(chunks)

    assert done is True
    assert any(p["type"] == "data-guardrail" for p in parts)
    text = "".join(p.get("delta", "") for p in parts if p["type"] == "text-delta")
    assert text == "请求已被安全策略拦截。"


class _FakeRunner:
    async def stream(self, run, *, resume=None):  # noqa: ANN001
        yield {"event": "start", "data": {"thread_id": run.thread_id}}
        yield {"event": "token", "data": {"delta": "ok"}}
        yield {"event": "done", "data": {"thread_id": run.thread_id, "status": "ok", "citations": []}}


@pytest.mark.asyncio
async def test_ui_stream_endpoint_sets_protocol_header():
    app = FastAPI()
    app.include_router(chat.router)
    app.state.runner = _FakeRunner()
    app.dependency_overrides[get_caller] = lambda: CallerContext(
        user_id="E1024",
        tenant_id="ten_a",
        roles=["employee"],
        scopes=["chat:write"],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/ui-stream", json={"message": "你好"})

    assert response.status_code == 200
    assert response.headers[STREAM_HEADER] == STREAM_VERSION
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "text-delta" in response.text
    assert response.text.rstrip().endswith("[DONE]")
