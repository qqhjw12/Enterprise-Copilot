"""图状态与运行时依赖。

状态里只放"可序列化"的东西，否则 PostgreSQL checkpointer 存不进去
（这是很多人第一次上 PostgresSaver 时踩的坑：往 state 里塞了一个工具对象）。

运行时依赖（LLM 网关、工具箱、调用者身份）通过 config["configurable"]["deps"] 注入，
不进 state、不进 checkpoint —— 它们是"进程内的能力"，不是"会话的记忆"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from runtime.config import RuntimeSettings
from runtime.llm.gateway import LLMGateway
from runtime.mcp_layer.client import _BaseToolbox
from runtime.mcp_layer.schemas import CallerContext

Intent = Literal["knowledge", "operation"]
RunStatus = Literal["running", "ok", "fallback", "awaiting_approval"]


class AgentState(TypedDict, total=False):
    # 会话历史（add_messages 负责追加与按 id 覆盖）
    messages: Annotated[list[AnyMessage], add_messages]

    # 身份
    tenant_id: str
    user_id: str
    thread_id: str
    roles: list[str]
    scopes: list[str]

    # 路由与检索
    intent: Intent
    citations: list[dict[str, Any]]
    retrieved_block: str

    # 工具循环
    pending_tool_calls: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    iterations: int
    consecutive_failures: int
    approvals: list[dict[str, Any]]

    # 结果
    answer: str
    status: RunStatus
    fallback_reason: str | None
    degraded: bool
    usage: dict[str, Any]


@dataclass(slots=True)
class GraphDeps:
    """一次运行期间共享的能力集合。"""

    llm: LLMGateway
    toolbox: _BaseToolbox
    caller: CallerContext
    settings: RuntimeSettings
    rag_top_k: int = 5

    def tool_schemas(self) -> list[dict[str, Any]]:
        """转成 OpenAI tools 格式。caller 不在 schema 里 —— 模型无权决定自己是谁。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self.toolbox.tool_specs()
        ]


def get_deps(config: dict[str, Any]) -> GraphDeps:
    deps = (config.get("configurable") or {}).get("deps")
    if deps is None:
        raise RuntimeError("图缺少运行时依赖：请通过 config.configurable.deps 注入 GraphDeps")
    return deps


def initial_state(
    *,
    message: str,
    caller: CallerContext,
    thread_id: str,
) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=message)],
        tenant_id=caller.tenant_id,
        user_id=caller.user_id,
        thread_id=thread_id,
        roles=list(caller.roles),
        scopes=list(caller.scopes),
        citations=[],
        retrieved_block="",
        pending_tool_calls=[],
        tool_events=[],
        iterations=0,
        consecutive_failures=0,
        approvals=[],
        answer="",
        status="running",
        fallback_reason=None,
        degraded=False,
        usage={},
    )
