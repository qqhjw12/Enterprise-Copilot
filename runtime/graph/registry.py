"""图注册表：把"项目 → 图"的映射从请求处理里摘出来。

新增一个 Agent 应用（如"财务对账助手"）= 新增一个 builder 并注册，
不需要改 API 层。这也是 Brescou/langgraph-agent-stack 里 Domain Pack 的思路。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from runtime.graph.builder import build_state_graph, compile_graph

GraphBuilder = Callable[[], Any]

_REGISTRY: dict[str, GraphBuilder] = {"copilot_default": build_state_graph}
_COMPILED: dict[str, CompiledStateGraph] = {}


def register(graph_key: str, builder: GraphBuilder) -> None:
    _REGISTRY[graph_key] = builder
    _COMPILED.pop(graph_key, None)


def list_graphs() -> list[str]:
    return sorted(_REGISTRY)


def get_compiled(graph_key: str, checkpointer: Any) -> CompiledStateGraph:
    """按 key 缓存已编译图（checkpointer 在进程内是单例）。"""
    key = graph_key if graph_key in _REGISTRY else "copilot_default"
    if key not in _COMPILED:
        _COMPILED[key] = compile_graph(checkpointer)
    return _COMPILED[key]


def reset_cache() -> None:
    _COMPILED.clear()
