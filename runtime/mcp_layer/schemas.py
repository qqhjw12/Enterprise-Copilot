"""MCP Server 的输入/输出契约。

所有工具统一返回 OpsResult 信封，而不是"成功返回 dict、失败抛异常"：
  - 越权 → ok=false, error.code=permission_denied（Agent 可读、可解释、可兜底）
  - 参数不合法 → ok=false, error.code=invalid_argument（由业务校验给出，非堆栈）
  - 业务不存在 → ok=false, error.code=xxx_not_found

这样上层（LangGraph 工具节点）不需要 try/except 各种异常类型，
只判断 ok 字段就能决定"重试 / 换参数 / 走兜底"。
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CallerContext(BaseModel):
    """调用方身份。由 Runtime 在调用 MCP 工具时注入，而不是让模型自己编。"""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=64)
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "*" in self.scopes or "admin" in self.roles


class OpsError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class OpsResult(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: OpsError | None = None

    @classmethod
    def success(cls, data: BaseModel | dict[str, Any]) -> OpsResult:
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        return cls(ok=True, data=payload)

    @classmethod
    def failure(cls, code: str, message: str, **details: Any) -> OpsResult:
        return cls(ok=False, error=OpsError(code=code, message=message, details=details))


# ---------------------------------------------------------------- 资产

class Asset(BaseModel):
    asset_id: str
    name: str
    category: str
    status: Literal["in_use", "idle", "repair", "lost"]
    assigned_at: date | None = None
    serial: str = ""


class GetEmployeeAssetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caller: CallerContext
    employee_id: str = Field(min_length=2, max_length=32, description="员工工号，如 E1024")


class EmployeeAssets(BaseModel):
    employee_id: str
    employee_name: str
    department: str
    assets: list[Asset]

    @property
    def count(self) -> int:
        return len(self.assets)


# ---------------------------------------------------------------- 请假

LeaveType = Literal["annual", "sick", "personal", "compensatory"]


class SubmitLeaveRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caller: CallerContext
    employee_id: str = Field(min_length=2, max_length=32)
    leave_type: LeaveType
    start_date: date
    end_date: date
    reason: str = Field(default="", max_length=500)
    idempotency_key: str = Field(default="", max_length=64, description="重试时的幂等键，避免重复提交")

    @field_validator("end_date")
    @classmethod
    def _check_range(cls, value: date, info) -> date:  # noqa: ANN001
        start = info.data.get("start_date")
        if start and value < start:
            raise ValueError("结束日期不能早于开始日期")
        return value


class LeaveRequestResult(BaseModel):
    request_id: str
    employee_id: str
    leave_type: str
    days: int
    status: Literal["pending_approval", "approved", "rejected"]
    submitted_at: str
    next_approver: str


# ---------------------------------------------------------------- 工单

TicketPriority = Literal["P1", "P2", "P3", "P4"]


class QueryTicketStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caller: CallerContext
    ticket_id: str = Field(min_length=3, max_length=32, description="工单号，如 T-2026-0042")


class TicketStatus(BaseModel):
    ticket_id: str
    title: str
    status: Literal["pending", "in_progress", "waiting_verify", "resolved", "closed"]
    priority: TicketPriority
    assignee: str
    requester_id: str
    created_at: str
    sla_due_at: str
    sla_breached: bool


class CreateTicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caller: CallerContext
    title: str = Field(min_length=4, max_length=120)
    description: str = Field(default="", max_length=2000)
    priority: TicketPriority = "P3"
    requester_id: str = Field(default="", max_length=32)
    idempotency_key: str = Field(default="", max_length=64)


class CreateTicketResult(BaseModel):
    ticket_id: str
    status: str
    priority: str
    sla_due_at: str
    assignee: str
    deduplicated: bool = False


# ---------------------------------------------------------------- 通讯录

class SearchDirectoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caller: CallerContext
    keyword: str = Field(min_length=1, max_length=64, description="姓名 / 部门 / 邮箱关键字")
    limit: int = Field(default=5, ge=1, le=20)


class DirectoryEntry(BaseModel):
    employee_id: str
    name: str
    department: str
    title: str
    email: str


class DirectoryResult(BaseModel):
    keyword: str
    entries: list[DirectoryEntry]
