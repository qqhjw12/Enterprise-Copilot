"""MCP 工具背后的"内部业务系统"。

用内存实现替代真实 OA/ITSM：接口形状（工号、工单号、SLA 计算）按真实系统建模，
替换成 HTTP 调用时无需改动 tool 层的 Schema 与错误码。
"""
from __future__ import annotations

import threading
import uuid
from datetime import date, datetime, timedelta

from runtime.mcp_layer.schemas import Asset, DirectoryEntry, LeaveRequestResult, TicketStatus

_LOCK = threading.Lock()

# 与 ops-ticket-sla.md 中的 SLA 保持一致
SLA_HOURS = {"P1": 2, "P2": 8, "P3": 48, "P4": 120}

_EMPLOYEES: dict[str, dict] = {
    "E1024": {"name": "张伟", "department": "研发中心", "title": "后端工程师", "email": "zhangwei@corp.example.com"},
    "E2048": {"name": "李静", "department": "人力资源部", "title": "HRBP", "email": "lijing@corp.example.com"},
    "E3001": {"name": "王强", "department": "财务部", "title": "财务分析师", "email": "wangqiang@corp.example.com"},
    "E4002": {"name": "陈曦", "department": "IT 服务部", "title": "运维工程师", "email": "chenxi@corp.example.com"},
}

_ASSETS: dict[str, list[dict]] = {
    "E1024": [
        {"asset_id": "AST-0001", "name": "MacBook Pro 14", "category": "laptop", "status": "in_use", "assigned_at": date(2025, 3, 1), "serial": "C02XK1AB"},
        {"asset_id": "AST-0002", "name": "Dell U2723QE 显示器", "category": "monitor", "status": "in_use", "assigned_at": date(2025, 3, 2), "serial": "CN0M1"},
    ],
    "E2048": [
        {"asset_id": "AST-0101", "name": "ThinkPad T14", "category": "laptop", "status": "in_use", "assigned_at": date(2024, 11, 5), "serial": "PF3XYZ"},
    ],
    "E4002": [
        {"asset_id": "AST-0201", "name": "ThinkPad T14", "category": "laptop", "status": "repair", "assigned_at": date(2024, 6, 1), "serial": "PF3ABC"},
        {"asset_id": "AST-0202", "name": "罗技 MX Master 3", "category": "peripheral", "status": "in_use", "assigned_at": date(2025, 1, 9), "serial": "LMX3301"},
    ],
}

_TICKETS: dict[str, dict] = {
    "T-2026-0042": {
        "title": "财务系统无法提交报销单",
        "status": "in_progress",
        "priority": "P2",
        "assignee": "陈曦",
        "requester_id": "E3001",
        "created_at": datetime(2026, 10, 28, 9, 12),
    },
    "T-2026-0039": {
        "title": "会议室投影无信号",
        "status": "resolved",
        "priority": "P3",
        "assignee": "陈曦",
        "requester_id": "E2048",
        "created_at": datetime(2026, 10, 20, 14, 30),
    },
}

# 幂等表：idempotency_key -> 已产生的结果
_LEAVE_REQUESTS: dict[str, dict] = {}
_IDEMPOTENT: dict[str, dict] = {}


def get_employee(employee_id: str) -> dict | None:
    return _EMPLOYEES.get(employee_id)


def list_assets(employee_id: str) -> list[Asset]:
    return [Asset(**item) for item in _ASSETS.get(employee_id, [])]


def search_directory(keyword: str, limit: int) -> list[DirectoryEntry]:
    kw = keyword.strip().lower()
    hits: list[DirectoryEntry] = []
    for emp_id, info in _EMPLOYEES.items():
        haystack = f"{emp_id} {info['name']} {info['department']} {info['title']} {info['email']}".lower()
        if kw in haystack:
            hits.append(DirectoryEntry(employee_id=emp_id, **info))
    return hits[:limit]


def get_ticket(ticket_id: str) -> dict | None:
    return _TICKETS.get(ticket_id.upper())


def ticket_view(ticket_id: str, record: dict) -> TicketStatus:
    created_at: datetime = record["created_at"]
    sla_due = created_at + timedelta(hours=SLA_HOURS[record["priority"]])
    closed = record["status"] in {"resolved", "closed"}
    return TicketStatus(
        ticket_id=ticket_id,
        title=record["title"],
        status=record["status"],
        priority=record["priority"],
        assignee=record["assignee"],
        requester_id=record["requester_id"],
        created_at=created_at.isoformat(),
        sla_due_at=sla_due.isoformat(),
        sla_breached=(not closed) and datetime.now() > sla_due,
    )


def create_leave_request(
    *,
    employee_id: str,
    leave_type: str,
    start_date: date,
    end_date: date,
    reason: str,
    idempotency_key: str,
) -> tuple[LeaveRequestResult, bool]:
    """返回 (结果, 是否命中幂等)。"""
    with _LOCK:
        if idempotency_key and idempotency_key in _IDEMPOTENT:
            return LeaveRequestResult(**_IDEMPOTENT[idempotency_key]), True

        days = (end_date - start_date).days + 1
        request_id = f"LV-{datetime.now():%Y%m}-{uuid.uuid4().hex[:5].upper()}"
        result = LeaveRequestResult(
            request_id=request_id,
            employee_id=employee_id,
            leave_type=leave_type,
            days=days,
            status="pending_approval",
            submitted_at=datetime.now().isoformat(timespec="seconds"),
            next_approver="直属主管（张伟）" if leave_type != "annual" else "直属主管（张伟）→ HR 备案",
        )
        _LEAVE_REQUESTS[request_id] = result.model_dump()
        if idempotency_key:
            _IDEMPOTENT[idempotency_key] = result.model_dump()
        return result, False


def create_ticket(
    *,
    title: str,
    description: str,
    priority: str,
    requester_id: str,
    idempotency_key: str,
) -> tuple[dict, bool]:
    with _LOCK:
        if idempotency_key and idempotency_key in _IDEMPOTENT:
            return _IDEMPOTENT[idempotency_key], True

        now = datetime.now()
        ticket_id = f"T-{now:%Y}-{len(_TICKETS) + 100:04d}"
        record = {
            "title": title,
            "status": "pending",
            "priority": priority,
            "assignee": "待分配",
            "requester_id": requester_id or "unknown",
            "created_at": now,
            "description": description,
        }
        _TICKETS[ticket_id] = record
        payload = {
            "ticket_id": ticket_id,
            "status": record["status"],
            "priority": priority,
            "sla_due_at": (now + timedelta(hours=SLA_HOURS[priority])).isoformat(timespec="seconds"),
            "assignee": record["assignee"],
            "deduplicated": False,
        }
        if idempotency_key:
            _IDEMPOTENT[idempotency_key] = payload
        return payload, False


def reset() -> None:
    """测试用：清空可变状态。"""
    with _LOCK:
        _LEAVE_REQUESTS.clear()
        _IDEMPOTENT.clear()
