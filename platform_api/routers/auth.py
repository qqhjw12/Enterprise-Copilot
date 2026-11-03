from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.errors import Forbidden, Unauthorized
from common.security import Principal, issue_access_token, verify_password
from platform_api.config import get_settings
from platform_api.db import get_db
from platform_api.deps import audit, get_principal
from platform_api.models import Tenant, User
from platform_api.schemas import LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    settings = get_settings()

    tenant = await db.scalar(select(Tenant).where(Tenant.slug == payload.tenant))
    if tenant is None or not tenant.is_active:
        # 统一话术，避免通过错误信息枚举租户
        await audit(
            db, action="auth.login", tenant_id=payload.tenant, outcome="failure",
            detail={"email": payload.email, "reason": "tenant_not_found"}, request=request,
        )
        await db.commit()
        raise Unauthorized("账号或密码错误")

    user = await db.scalar(
        select(User).where(User.tenant_id == tenant.id, User.email == payload.email)
    )
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        await audit(
            db, action="auth.login", tenant_id=tenant.id, outcome="failure",
            detail={"email": payload.email, "reason": "bad_credentials"}, request=request,
        )
        await db.commit()
        raise Unauthorized("账号或密码错误")

    user.last_login_at = datetime.now(UTC)
    principal = Principal(
        user_id=user.id, tenant_id=tenant.id, email=user.email,
        roles=list(user.roles or []), scopes=list(user.scopes or []),
        employee_no=user.employee_no or "",
    )
    await audit(db, action="auth.login", principal=principal, resource_type="user",
                resource_id=user.id, request=request)
    await db.commit()

    token = issue_access_token(
        user_id=user.id,
        tenant_id=tenant.id,
        email=user.email,
        roles=list(user.roles or []),
        scopes=list(user.scopes or []),
        employee_no=user.employee_no or "",
        ttl_minutes=settings.jwt_ttl_minutes,
        secret=settings.jwt_secret,
    )
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_ttl_minutes * 60,
        user=UserOut.model_validate(user),
    )


@router.get("/me", response_model=UserOut)
async def me(principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)) -> UserOut:
    user = await db.get(User, principal.user_id)
    if user is None:
        raise Forbidden("用户不存在")
    return UserOut.model_validate(user)
