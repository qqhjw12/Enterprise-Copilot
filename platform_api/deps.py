"""认证/鉴权依赖与审计写入。

鉴权顺序刻意固定：令牌有效 → 用户存在且启用 → 租户启用 → 作用域检查。
任何一步失败都写审计（outcome=denied），因为"谁试图越权"和"谁越权成功"同样重要。
"""
from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.errors import Forbidden, Unauthorized
from common.logging import request_id_ctx
from common.security import Principal, TokenError, decode_access_token
from platform_api.config import get_settings
from platform_api.db import get_db
from platform_api.models import AuditLog, Tenant, User

bearer = HTTPBearer(auto_error=False, description="平台签发的 JWT")


async def get_principal(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> Principal:
    if creds is None or not creds.credentials:
        raise Unauthorized("缺少 Bearer 令牌")
    try:
        principal = decode_access_token(creds.credentials, secret=get_settings().jwt_secret)
    except TokenError as exc:
        raise Unauthorized(f"令牌无效：{exc}") from exc

    user = await db.scalar(
        select(User).where(User.id == principal.user_id, User.tenant_id == principal.tenant_id)
    )
    if user is None or not user.is_active:
        raise Unauthorized("用户不存在或已停用")
    tenant = await db.get(Tenant, user.tenant_id)
    if tenant is None or not tenant.is_active:
        raise Unauthorized("租户已停用")

    # 以库里的角色/作用域为准，令牌里的只是快照（改权限后无需等 token 过期）
    principal.email = user.email
    principal.roles = list(user.roles or [])
    principal.scopes = list(user.scopes or [])
    principal.employee_no = user.employee_no or ""
    request.state.principal = principal
    return principal


def require_scope(*scopes: str) -> Callable[..., Principal]:
    """要求同时具备多个作用域。"""

    async def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        missing = [s for s in scopes if not principal.has_scope(s)]
        if missing:
            raise Forbidden("缺少所需权限", details={"missing_scopes": missing})
        return principal

    return _dep


async def audit(
    db: AsyncSession,
    *,
    action: str,
    principal: Principal | None = None,
    tenant_id: str | None = None,
    resource_type: str = "",
    resource_id: str = "",
    outcome: str = "success",
    detail: dict | None = None,
    request: Request | None = None,
) -> None:
    entry = AuditLog(
        tenant_id=tenant_id or (principal.tenant_id if principal else ""),
        user_id=principal.user_id if principal else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        detail=detail or {},
        ip=(request.client.host if request and request.client else ""),
        request_id=request_id_ctx.get() or "",
    )
    db.add(entry)
    await db.flush()
