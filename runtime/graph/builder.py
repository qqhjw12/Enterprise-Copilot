"""图装配：条件路由 / 工具循环 / HITL / 兜底。

拓扑（不是线性链）：

    START → route ─┬─ knowledge → retrieve → answer → END
                   │                 └→ fallback → END
                   └─ operation → agent ─┬─ 有写操作 → approval ─┬─ approve/edit → tools
                                         │                      └─ reject      → END
                                         ├─ 只有读操作 ────────→ tools
                                         ├─ 无工具调用 → END
                                         └─ 预算/模型异常 → fallback → END

    tools ─┬─ 正常 → agent（回到模型，最多 max_iterations 轮）
           └─ 连续失败 / 超轮次 → fallback → END
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from runtime.config import get_settings
from runtime.graph import nodes
from runtime.graph.state import AgentState


def route_to_branch(state: AgentState) -> str:
    return "operation" if state.get("intent") == "operation" else "knowledge"


def after_retrieve(state: AgentState) -> str:
    return "fallback" if state.get("fallback_reason") else "answer"


def after_answer(state: AgentState) -> str:
    return "fallback" if state.get("fallback_reason") else END


def after_agent(state: AgentState) -> str:
    if state.get("fallback_reason"):
        return "fallback"
    pending = state.get("pending_tool_calls") or []
    if not pending:
        return END
    # 只要这一批里有写操作，就必须先过审批；读操作直接执行
    if any(bool(call.get("write")) for call in pending):
        return "approval"
    return "tools"


def after_approval(state: AgentState) -> str:
    if state.get("pending_tool_calls"):
        return "tools"          # 批准 / 修改后照常执行
    if state.get("answer"):
        return END               # 拒绝：已有终止答复，不再回到模型
    return "agent"


def after_tools(state: AgentState) -> str:
    return "fallback" if state.get("fallback_reason") else "agent"


def build_state_graph() -> StateGraph:
    """构建未编译的图。编译（绑定 checkpointer）在应用启动时做一次。"""
    graph = StateGraph(AgentState)

    graph.add_node("route", nodes.route_node)
    graph.add_node("knowledge_retrieve", nodes.knowledge_retrieve_node)
    graph.add_node("knowledge_answer", nodes.knowledge_answer_node)
    graph.add_node("agent", nodes.agent_node)
    graph.add_node("approval", nodes.approval_node)
    graph.add_node("tools", nodes.tools_node)
    graph.add_node("fallback", nodes.fallback_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route", route_to_branch, {"knowledge": "knowledge_retrieve", "operation": "agent"}
    )
    graph.add_conditional_edges(
        "knowledge_retrieve", after_retrieve, {"answer": "knowledge_answer", "fallback": "fallback"}
    )
    graph.add_conditional_edges(
        "knowledge_answer", after_answer, {"fallback": "fallback", END: END}
    )
    graph.add_conditional_edges(
        "agent",
        after_agent,
        {"approval": "approval", "tools": "tools", "fallback": "fallback", END: END},
    )
    graph.add_conditional_edges(
        "approval", after_approval, {"tools": "tools", "agent": "agent", END: END}
    )
    graph.add_conditional_edges(
        "tools", after_tools, {"agent": "agent", "fallback": "fallback"}
    )
    graph.add_edge("fallback", END)

    return graph


def compile_graph(checkpointer: Any = None, *, interrupt_before: list[str] | None = None):
    """编译图。

    interrupt_before 一般不用：HITL 是在 approval 节点内部用 interrupt() 实现的，
    比"在节点前中断"更灵活（可以在中断负载里带上具体参数给人看）。
    """
    settings = get_settings()
    build = build_state_graph()
    _ = settings
    return build.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
