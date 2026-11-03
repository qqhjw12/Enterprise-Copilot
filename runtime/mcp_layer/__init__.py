"""MCP 标准化工具层。

Server 端（server.py + schemas.py + backends.py）与 Client 端（client.py）分离，
中间由 MCP 协议连接；韧性策略（resilience.py）与治理表（policies.py）为两端共享。
"""
from runtime.mcp_layer.client import (
    LocalToolbox,
    McpToolbox,
    ToolCallOutcome,
    ToolSpec,
    build_tool_specs,
    build_toolbox,
)

__all__ = [
    "LocalToolbox",
    "McpToolbox",
    "ToolCallOutcome",
    "ToolSpec",
    "build_tool_specs",
    "build_toolbox",
]
