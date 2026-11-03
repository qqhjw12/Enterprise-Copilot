"""InternalOpsServer —— 把内部业务系统封装成标准 MCP Server。

三种启动方式：
  http  : python -m runtime.mcp_layer.server --transport http --port 8765   （compose 用）
  stdio : python -m runtime.mcp_layer.server --transport stdio               （本地单进程用）
  import: from runtime.mcp_layer.server import call_tool_sync                （单测直连）

鉴权在 Server 端做，而不是"信任 Client 传的 user_id"：
caller 由 Runtime 从已校验的 JWT 里解出来再注入；模型无法伪造别人的 scope
（它最多能改参数字符串，改不了 caller 对象 —— 因为 caller 不在模型可见的 tool schema 里）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from common.logging import setup_logging
from runtime.mcp_layer import backends
from runtime.mcp_layer.policies import TOOL_POLICIES
from runtime.mcp_layer.schemas import (
    CallerContext,
    CreateTicketInput,
    CreateTicketResult,
    DirectoryResult,
    EmployeeAssets,
    GetEmployeeAssetsInput,
    OpsResult,
    QueryTicketStatusInput,
    SearchDirectoryInput,
    SubmitLeaveRequestInput,
)

logger = logging.getLogger(__name__)

# MCP 2025-06-18 ToolAnnotations：让客户端在不理解业务的前提下也能判断风险
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)

mcp = FastMCP(
    name="internal-ops",
    instructions=(
        "企业内部运营工具集：查询员工资产、提交请假、查询/创建 IT 工单、检索通讯录。"
        "所有写操作都会被上层 Graph 拦截并要求人工审批。"
    ),
    host="0.0.0.0",
    port=8765,
)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request):  # noqa: ANN001
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "service": "mcp-ops", "tools": len(TOOL_POLICIES)})


# 可以看到"别人"数据的角色。注意这是**角色**而不是 scope：
# scope 决定"能不能用这个工具"，角色决定"能看到谁的数据"，两者不能混谈。
PRIVILEGED_ROLES = {"admin", "hr", "it"}


def _is_privileged(caller: CallerContext) -> bool:
    return "*" in caller.scopes or bool(set(caller.roles) & PRIVILEGED_ROLES)


def _guard(caller: CallerContext, tool_name: str) -> OpsResult | None:
    """统一的 scope 校验。返回 None 表示通过。"""
    policy = TOOL_POLICIES.get(tool_name)
    if policy is None:
        return OpsResult.failure("unknown_tool", f"未注册的工具：{tool_name}")
    if not caller.has_scope(policy.scope):
        logger.warning(
            "mcp_permission_denied",
            extra={"tool": tool_name, "user_id": caller.user_id, "required": policy.scope},
        )
        return OpsResult.failure(
            "permission_denied",
            f"当前账号无权调用「{policy.label}」",
            required_scope=policy.scope,
            user_scopes=caller.scopes,
        )
    return None


# ---------------------------------------------------------------- 只读工具

@mcp.tool(name="get_employee_assets", title="查询员工资产", annotations=_READ_ONLY)
async def get_employee_assets(payload: GetEmployeeAssetsInput) -> OpsResult:
    """查询员工名下的固定资产（笔记本、显示器、外设）及状态。"""
    if (denied := _guard(payload.caller, "get_employee_assets")) is not None:
        return denied
    # 有 ops:asset:read 不等于"能查任何人的资产"：员工只能查自己，
    # 否则一句"帮我查 E2048 的资产"就能横向越权（真实事故里最常见的一条）。
    if payload.employee_id != payload.caller.user_id and not _is_privileged(payload.caller):
        return OpsResult.failure(
            "permission_denied",
            "只能查询本人名下的资产；查询他人资产需 HR/IT 角色",
            requested=payload.employee_id,
            operator=payload.caller.user_id,
        )

    employee = backends.get_employee(payload.employee_id)
    if employee is None:
        return OpsResult.failure("employee_not_found", f"工号 {payload.employee_id} 不存在")

    assets = backends.list_assets(payload.employee_id)
    return OpsResult.success(
        EmployeeAssets(
            employee_id=payload.employee_id,
            employee_name=employee["name"],
            department=employee["department"],
            assets=assets,
        )
    )


@mcp.tool(name="query_ticket_status", title="查询工单状态", annotations=_READ_ONLY)
async def query_ticket_status(payload: QueryTicketStatusInput) -> OpsResult:
    """按工单号查询 IT 工单的处理状态、负责人与 SLA 到期时间。"""
    if (denied := _guard(payload.caller, "query_ticket_status")) is not None:
        return denied

    record = backends.get_ticket(payload.ticket_id)
    if record is None:
        return OpsResult.failure("ticket_not_found", f"工单 {payload.ticket_id} 不存在")
    return OpsResult.success(backends.ticket_view(payload.ticket_id.upper(), record))


@mcp.tool(name="search_directory", title="检索通讯录", annotations=_READ_ONLY)
async def search_directory(payload: SearchDirectoryInput) -> OpsResult:
    """按姓名/部门/邮箱关键字检索内部通讯录（仅返回公开字段）。"""
    if (denied := _guard(payload.caller, "search_directory")) is not None:
        return denied

    entries = backends.search_directory(payload.keyword, payload.limit)
    if not entries:
        return OpsResult.failure("no_match", f"未找到与「{payload.keyword}」匹配的同事")
    return OpsResult.success(DirectoryResult(keyword=payload.keyword, entries=entries))


# ---------------------------------------------------------------- 写操作（走 HITL）

@mcp.tool(name="submit_leave_request", title="提交请假申请", annotations=_WRITE)
async def submit_leave_request(payload: SubmitLeaveRequestInput) -> OpsResult:
    """提交请假申请（写操作）。支持 idempotency_key 防重复提交。"""
    if (denied := _guard(payload.caller, "submit_leave_request")) is not None:
        return denied

    if payload.employee_id != payload.caller.user_id and "hr" not in payload.caller.roles:
        # 代他人请假属于高敏操作，限定 HR 才能做
        return OpsResult.failure(
            "permission_denied",
            "只能为本人提交请假申请；代他人提交需要 HR 角色",
            operator=payload.caller.user_id,
            target=payload.employee_id,
        )

    result, deduplicated = backends.create_leave_request(
        employee_id=payload.employee_id,
        leave_type=payload.leave_type,
        start_date=payload.start_date,
        end_date=payload.end_date,
        reason=payload.reason,
        idempotency_key=payload.idempotency_key,
    )
    data = result.model_dump()
    data["deduplicated"] = deduplicated
    return OpsResult.success(data)


@mcp.tool(name="create_ticket", title="创建 IT 工单", annotations=_WRITE)
async def create_ticket(payload: CreateTicketInput) -> OpsResult:
    """创建 IT 工单（写操作）。P1/P2 会自动通知值班经理。"""
    if (denied := _guard(payload.caller, "create_ticket")) is not None:
        return denied

    data, deduplicated = backends.create_ticket(
        title=payload.title,
        description=payload.description,
        priority=payload.priority,
        requester_id=payload.requester_id or payload.caller.user_id,
        idempotency_key=payload.idempotency_key,
    )
    result = CreateTicketResult(deduplicated=deduplicated, **{k: v for k, v in data.items() if k != "deduplicated"})
    return OpsResult.success(result)


async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """进程内直调（测试 / 无 MCP 容器时的降级路径）。返回与 MCP 一致的 JSON 结构。"""
    result = await mcp.call_tool(name, arguments)
    # FastMCP.call_tool 返回 (content_blocks, structured) 或已解包的结果，按版本兼容处理
    if isinstance(result, tuple):  # (contents, structured_content)
        contents, structured = result
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        if isinstance(structured, dict):
            return structured
        for block in contents or []:
            text = getattr(block, "text", None)
            if text:
                import json

                return json.loads(text)
        return {"ok": False, "error": {"code": "empty_result", "message": "工具未返回内容", "details": {}}}
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="InternalOpsServer (MCP)")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "http", "sse"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    setup_logging("mcp-ops", stream=sys.stderr)  # stdio 模式下 stdout 是协议通道
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    transport = "streamable-http" if args.transport == "http" else args.transport
    logger.info("mcp_server_starting", extra={"transport": transport, "port": args.port})
    mcp.run(transport=transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
