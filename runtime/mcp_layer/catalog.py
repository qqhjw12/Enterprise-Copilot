"""工具目录（Tool Catalog）：把 MCP 契约以可 diff 的 JSON 暴露给平台与 CI。

企业里工具不是"模型能看见就行"，还要回答四个问题：
  1. 这个工具改了参数，谁会被影响？ → schema_hash + 快照测试
  2. 它是不是写操作？要不要审批？ → annotations.readOnlyHint / policies
  3. 它读的是谁的数据？ → scope + 角色校验（在 Server 端）
  4. 它现在能不能用？ → transport / degraded / breaker 状态
"""
from __future__ import annotations

from typing import Any

from runtime.mcp_layer.client import TOOL_CONTRACT_VERSION, ToolSpec, build_tool_specs


def _tool_entry(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "scope": spec.scope,
        "write": spec.write,
        "annotations": spec.annotations,
        "input_schema": spec.parameters,
        "output_schema": spec.output_schema,
        "schema_hash": spec.schema_hash,
    }


def build_tool_catalog() -> dict[str, Any]:
    """纯函数：端点与快照测试共用，保证"线上看到的"就是"测试冻结的"。"""
    specs = build_tool_specs()
    return {
        "server": "internal-ops",
        "contract_version": TOOL_CONTRACT_VERSION,
        "protocol": "mcp",
        "spec": "2025-06-18",
        "tools": [_tool_entry(specs[name]) for name in sorted(specs)],
    }
