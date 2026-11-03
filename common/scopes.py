"""平台级常量：作用域与角色。

作用域是跨服务契约（控制面签发、MCP Server 校验），因此放在 common 下，
避免出现"MCP 认识的 scope 平台不认识"这类漂移。
"""
from __future__ import annotations

# 数据面
SCOPE_CHAT_READ = "chat:read"
SCOPE_CHAT_WRITE = "chat:write"
SCOPE_RAG_READ = "rag:read"
SCOPE_RAG_INGEST = "rag:ingest"
# 特批作用域：跨过 classification 等级读机密文档（薪酬/并购等），只有 HR/合规等岗位才该有
SCOPE_RAG_CONFIDENTIAL = "rag:confidential"

# 业务工具（MCP）
SCOPE_OPS_ASSET_READ = "ops:asset:read"
SCOPE_OPS_LEAVE_WRITE = "ops:leave:write"
SCOPE_OPS_TICKET_READ = "ops:ticket:read"
SCOPE_OPS_TICKET_WRITE = "ops:ticket:write"
SCOPE_OPS_DIRECTORY_READ = "ops:directory:read"

# 治理面
SCOPE_PROJECT_ADMIN = "project:admin"
SCOPE_MODEL_ADMIN = "model:admin"
SCOPE_AUDIT_READ = "audit:read"

ALL_SCOPES: tuple[str, ...] = (
    SCOPE_CHAT_READ,
    SCOPE_CHAT_WRITE,
    SCOPE_RAG_READ,
    SCOPE_RAG_INGEST,
    SCOPE_RAG_CONFIDENTIAL,
    SCOPE_OPS_ASSET_READ,
    SCOPE_OPS_LEAVE_WRITE,
    SCOPE_OPS_TICKET_READ,
    SCOPE_OPS_TICKET_WRITE,
    SCOPE_OPS_DIRECTORY_READ,
    SCOPE_PROJECT_ADMIN,
    SCOPE_MODEL_ADMIN,
    SCOPE_AUDIT_READ,
)

ROLE_SCOPES: dict[str, tuple[str, ...]] = {
    "admin": ("*",),
    "employee": (
        SCOPE_CHAT_READ,
        SCOPE_CHAT_WRITE,
        SCOPE_RAG_READ,
        SCOPE_OPS_ASSET_READ,
        SCOPE_OPS_TICKET_READ,
        SCOPE_OPS_TICKET_WRITE,
        SCOPE_OPS_LEAVE_WRITE,
        SCOPE_OPS_DIRECTORY_READ,
    ),
    # 只读审计岗：能查知识库与工单，不能发起写操作
    "auditor": (SCOPE_CHAT_READ, SCOPE_CHAT_WRITE, SCOPE_RAG_READ, SCOPE_OPS_TICKET_READ, SCOPE_AUDIT_READ),
    # HR：员工能力 + 机密文档 + 代他人提交请假（写操作仍需过 HITL 审批）
    "hr": (
        SCOPE_CHAT_READ,
        SCOPE_CHAT_WRITE,
        SCOPE_RAG_READ,
        SCOPE_RAG_INGEST,
        SCOPE_OPS_ASSET_READ,
        SCOPE_OPS_TICKET_READ,
        SCOPE_OPS_LEAVE_WRITE,
        SCOPE_OPS_DIRECTORY_READ,
        SCOPE_RAG_CONFIDENTIAL,
    ),
    "viewer": (SCOPE_CHAT_READ, SCOPE_CHAT_WRITE, SCOPE_RAG_READ),
}


def scopes_for_roles(roles: list[str]) -> list[str]:
    resolved: list[str] = []
    for role in roles:
        for scope in ROLE_SCOPES.get(role, ()):
            if scope not in resolved:
                resolved.append(scope)
    return resolved
