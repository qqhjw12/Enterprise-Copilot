from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.errors import Conflict
from common.scopes import SCOPE_PROJECT_ADMIN, scopes_for_roles
from common.security import Principal, hash_password
from platform_api.db import get_db
from platform_api.deps import audit, require_scope
from platform_api.models import User
from platform_api.schemas import UserCreate, UserOut

router = APIRouter(prefix="/v1/users", tags=["users"])


@router.get("", response_model=list[UserOut])
async def list_users(
    principal: Principal = Depends(require_scope(SCOPE_PROJECT_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> list[UserOut]:
    rows = await db.scalars(select(User).where(User.tenant_id == principal.tenant_id).order_by(User.created_at))
    return [UserOut.model_validate(u) for u in rows]


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    principal: Principal = Depends(require_scope(SCOPE_PROJECT_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> UserOut:
    exists = await db.scalar(
        select(User).where(User.tenant_id == principal.tenant_id, User.email == payload.email)
    )
    if exists:
        raise Conflict("该邮箱已存在")

    roles = payload.roles or ["employee"]
    user = User(
        tenant_id=principal.tenant_id,
        email=payload.email,
        display_name=payload.display_name or payload.email.split("@")[0],
        employee_no=payload.employee_no,
        password_hash=hash_password(payload.password),
        roles=roles,
        scopes=payload.scopes if payload.scopes is not None else scopes_for_roles(roles),
    )
    db.add(user)
    await db.flush()
    await audit(db, action="user.create", principal=principal, resource_type="user",
                resource_id=user.id, detail={"email": user.email, "roles": roles})
    await db.commit()
    return UserOut.model_validate(user)
