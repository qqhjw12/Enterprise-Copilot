"""LangGraph 状态图测试：路由 / 工具循环 / HITL / 兜底 / 预算。

全部用 ScriptedChat 驱动，不触网、不花钱，路径可精确断言。
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from runtime.config import get_settings
from runtime.graph.builder import compile_graph
from runtime.graph.nodes import classify_intent
from runtime.graph.state import GraphDeps, initial_state
from runtime.llm.client import LLMResponse, ScriptedChat, ToolCall
from runtime.llm.gateway import LLMGateway
from runtime.mcp_layer import backends
from runtime.mcp_layer.client import LocalToolbox
from runtime.mcp_layer.schemas import CallerContext
from runtime.obs.budget import BudgetGuard


def make_deps(responses: list[LLMResponse], caller: CallerContext, *, limit_usd: float = 0.05, spent: float = 0.0) -> GraphDeps:
    client = ScriptedChat(responses)
    budget = BudgetGuard(limit_usd=limit_usd, spent_usd=spent)
    gateway = LLMGateway(client, primary_model="scripted", cheap_model="scripted", budget=budget)
    return GraphDeps(llm=gateway, toolbox=LocalToolbox(), caller=caller, settings=get_settings())


def tool_response(name: str, args: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=args)],
        model="scripted",
        finish_reason="tool_calls",
    )


@pytest.fixture
def graph():
    return compile_graph(InMemorySaver())


# ---------------------------------------------------------------- 路由

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("年假有几天", "knowledge"),
        ("请假流程是什么", "knowledge"),
        ("报销标准是多少", "knowledge"),
        ("帮我查一下我的资产", "operation"),
        ("帮我提交年假申请 2026-11-10 到 2026-11-12", "operation"),
        ("工单 T-2026-0042 什么状态", "operation"),
    ],
)
def test_intent_classification(text, expected):
    assert classify_intent(text) == expected


# ---------------------------------------------------------------- 知识路径

@pytest.mark.asyncio
async def test_knowledge_path_returns_citations(graph, caller):
    deps = make_deps([LLMResponse(content="年假为 7~15 天，按工龄计算 [1]。", model="scripted")], caller)
    config = {"configurable": {"thread_id": "t-knowledge", "deps": deps}}
    state = await graph.ainvoke(initial_state(message="年假有几天", caller=caller, thread_id="t-knowledge"), config)

    assert state["intent"] == "knowledge"
    assert state["status"] == "ok"
    assert state["citations"], "知识路径必须带引用"
    assert "年假" in state["answer"]


@pytest.mark.asyncio
async def test_knowledge_path_without_hits_skips_llm(graph, caller, monkeypatch):
    """没有检索命中时不应该调用模型：省一次费用，也断掉"无依据硬编"的可能。

    这里直接打桩检索结果，而不是依赖某句"恰好检索不到"的问法 ——
    后者会随分词/阈值调整而失效，属于脆弱的测试。
    """
    from runtime.graph import nodes as graph_nodes
    from runtime.rag.types import RetrievedContext

    async def empty_retrieve(*_args, **_kwargs):
        return RetrievedContext(chunks=[])

    monkeypatch.setattr(graph_nodes, "retrieve", empty_retrieve)

    deps = make_deps([LLMResponse(content="不应被调用", model="scripted")], caller)
    config = {"configurable": {"thread_id": "t-miss", "deps": deps}}
    state = await graph.ainvoke(
        initial_state(message="公司有没有太空电梯补贴", caller=caller, thread_id="t-miss"), config
    )
    assert state["status"] == "ok"
    assert state["citations"] == []
    assert "未收录" in state["answer"]
    assert deps.llm.client.calls == [], "没有命中时不应该调用模型"


# ---------------------------------------------------------------- 工具路径

@pytest.mark.asyncio
async def test_read_tool_executes_without_approval(graph, caller):
    deps = make_deps(
        [
            tool_response("get_employee_assets", {"employee_id": "E1024"}),
            LLMResponse(content="你名下有 2 件资产。", model="scripted"),
        ],
        caller,
    )
    config = {"configurable": {"thread_id": "t-read", "deps": deps}}
    state = await graph.ainvoke(initial_state(message="帮我查一下我的资产", caller=caller, thread_id="t-read"), config)

    assert state["status"] == "ok"
    assert state["answer"] == "你名下有 2 件资产。"
    assert state["tool_events"] and state["tool_events"][0]["ok"] is True
    assert state["iterations"] == 2


@pytest.mark.asyncio
async def test_write_tool_interrupts_then_executes_once(graph, caller):
    backends.reset()
    deps = make_deps(
        [
            tool_response(
                "submit_leave_request",
                {
                    "employee_id": "E1024",
                    "leave_type": "annual",
                    "start_date": "2026-11-10",
                    "end_date": "2026-11-12",
                },
            ),
            LLMResponse(content="请假申请已提交。", model="scripted"),
        ],
        caller,
    )
    config = {"configurable": {"thread_id": "t-write", "deps": deps}}
    first = await graph.ainvoke(
        initial_state(message="帮我提交年假申请 2026-11-10 到 2026-11-12", caller=caller, thread_id="t-write"), config
    )

    # 第一次执行必须停在审批点，且没有产生任何写操作
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("approval",)
    assert backends._LEAVE_REQUESTS == {}
    assert first.get("pending_tool_calls"), "中断时应保留待执行调用"

    # 批准后继续执行
    final = await graph.ainvoke(Command(resume={"action": "approve"}), config)
    assert final["status"] == "ok"
    assert final["answer"] == "请假申请已提交。"
    assert len(backends._LEAVE_REQUESTS) == 1, "写操作必须只执行一次"


@pytest.mark.asyncio
async def test_write_tool_rejected_does_not_execute(graph, caller):
    backends.reset()
    deps = make_deps(
        [
            tool_response(
                "submit_leave_request",
                {"employee_id": "E1024", "leave_type": "annual", "start_date": "2026-12-01", "end_date": "2026-12-02"},
            ),
            LLMResponse(content="好的，已取消该申请。", model="scripted"),
        ],
        caller,
    )
    config = {"configurable": {"thread_id": "t-reject", "deps": deps}}
    await graph.ainvoke(initial_state(message="帮我提交年假申请 2026-12-01 到 2026-12-02", caller=caller, thread_id="t-reject"), config)
    final = await graph.ainvoke(Command(resume={"action": "reject"}), config)

    assert final["status"] == "ok"
    assert backends._LEAVE_REQUESTS == {}, "被拒绝的写操作不得落库"
    # 拒绝即终止：话术由系统给，不再回到模型。
    # 否则模型看到"工具报错"很可能换个参数再请求一次，等于绕过刚被拒绝的审批。
    assert "取消" in final["answer"] and "未在内部系统" in final["answer"]
    assert len(deps.llm.client.calls) == 1, "拒绝后不应再调用模型"


@pytest.mark.asyncio
async def test_consecutive_tool_failures_trigger_fallback(graph, caller):
    """连续两轮工具全失败 → 兜底话术，而不是把异常抛给用户。"""
    bad_call = tool_response("no_such_tool", {})
    deps = make_deps([bad_call], caller)  # 脚本复用同一条：每轮都请求不存在的工具
    config = {"configurable": {"thread_id": "t-fail", "deps": deps}}
    state = await graph.ainvoke(initial_state(message="帮我查一下我的资产", caller=caller, thread_id="t-fail"), config)

    assert state["status"] == "fallback"
    assert "系统响应异常" in state["answer"] or "已记录" in state["answer"]


@pytest.mark.asyncio
async def test_iteration_cap_degrades_gracefully(graph, caller):
    """一直请求工具时，超过 max_iterations 后不再执行工具，用已有结果收尾。"""
    call = tool_response("query_ticket_status", {"ticket_id": "T-2026-0042"})
    deps = make_deps([call], caller)
    config = {"configurable": {"thread_id": "t-cap", "deps": deps}}
    state = await graph.ainvoke(initial_state(message="帮我查一下我的资产", caller=caller, thread_id="t-cap"), config)

    assert state["iterations"] > get_settings().max_iterations
    assert state["status"] == "ok"
    assert state["degraded"] is True
    assert state["answer"]


@pytest.mark.asyncio
async def test_budget_exhausted_goes_to_fallback(graph, caller):
    deps = make_deps([LLMResponse(content="不该执行", model="scripted")], caller, limit_usd=0.01, spent=0.01)
    config = {"configurable": {"thread_id": "t-budget", "deps": deps}}
    state = await graph.ainvoke(initial_state(message="帮我查一下我的资产", caller=caller, thread_id="t-budget"), config)

    assert state["status"] == "fallback"
    assert "预算" in state["answer"]


@pytest.mark.asyncio
async def test_state_is_checkpointed_and_resumable_after_restart(graph, caller):
    """同一 thread_id 的第二次运行能看到上一轮历史（模拟 Worker 重启后恢复）。"""
    deps = make_deps([LLMResponse(content="第一轮回答", model="scripted")], caller)
    config = {"configurable": {"thread_id": "t-resume", "deps": deps}}
    await graph.ainvoke(initial_state(message="帮我查一下我的资产", caller=caller, thread_id="t-resume"), config)

    snapshot = await graph.aget_state(config)
    assert snapshot.values["answer"] == "第一轮回答"
    assert len(snapshot.values["messages"]) >= 2
