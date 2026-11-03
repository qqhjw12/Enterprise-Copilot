"""工具治理策略：一份表同时被 Server（鉴权）、Client（重试策略）、Graph（HITL 判定）使用。

把"哪些工具是写操作、需要什么 scope"集中在一处，
避免出现"Server 认为是写操作、图以为是只读、于是跳过了审批"这类跨模块不一致。
"""
from __future__ import annotations

from dataclasses import dataclass

from common.scopes import (
    SCOPE_OPS_ASSET_READ,
    SCOPE_OPS_DIRECTORY_READ,
    SCOPE_OPS_LEAVE_WRITE,
    SCOPE_OPS_TICKET_READ,
    SCOPE_OPS_TICKET_WRITE,
)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    name: str
    scope: str
    write: bool
    label: str
    # HITL 审批卡片里展示给人看的一句话
    approval_hint: str


TOOL_POLICIES: dict[str, ToolPolicy] = {
    "get_employee_assets": ToolPolicy(
        name="get_employee_assets",
        scope=SCOPE_OPS_ASSET_READ,
        write=False,
        label="查询员工资产",
        approval_hint="读取员工名下固定资产清单",
    ),
    "query_ticket_status": ToolPolicy(
        name="query_ticket_status",
        scope=SCOPE_OPS_TICKET_READ,
        write=False,
        label="查询工单状态",
        approval_hint="读取 IT 工单的处理进度与 SLA",
    ),
    "search_directory": ToolPolicy(
        name="search_directory",
        scope=SCOPE_OPS_DIRECTORY_READ,
        write=False,
        label="检索通讯录",
        approval_hint="按姓名/部门检索内部通讯录（仅返回公开字段）",
    ),
    "submit_leave_request": ToolPolicy(
        name="submit_leave_request",
        scope=SCOPE_OPS_LEAVE_WRITE,
        write=True,
        label="提交请假申请",
        approval_hint="将代表你在 OA 中正式发起请假流程，提交后需走审批",
    ),
    "create_ticket": ToolPolicy(
        name="create_ticket",
        scope=SCOPE_OPS_TICKET_WRITE,
        write=True,
        label="创建 IT 工单",
        approval_hint="将在 IT 服务台创建工单并按优先级排期",
    ),
}


def policy_for(tool_name: str) -> ToolPolicy | None:
    return TOOL_POLICIES.get(tool_name)


def is_write_tool(tool_name: str) -> bool:
    policy = TOOL_POLICIES.get(tool_name)
    return bool(policy and policy.write)


def required_scope(tool_name: str) -> str | None:
    policy = TOOL_POLICIES.get(tool_name)
    return policy.scope if policy else None
