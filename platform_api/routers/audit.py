from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from common.scopes import SCOPE_AUDIT_READ
from common.security import Principal
from platform_api.db import get_db
from platform_api.deps import require_scope
from platform_api.models import AuditLog
from platform_api.schemas import AuditPage

router = APIRouter(prefix="/v1/audit", tags=["audit"])


@router.get("", response_model=AuditPage)
async def list_audit(
    action: str | None = Query(default=None, description="按动作前缀过滤，如 auth."),
    user_id: str | None = None,
    outcome: str | None = Query(default=None, pattern="^(success|failure|denied)$"),
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_scope(SCOPE_AUDIT_READ)),
    db: AsyncSession = Depends(get_db),
) -> AuditPage:
    stmt = select(AuditLog).where(AuditLog.tenant_id == principal.tenant_id)
    if action:
        stmt = stmt.where(AuditLog.action.startswith(action))
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)
    if outcome:
        stmt = stmt.where(AuditLog.outcome == outcome)
    if since:
        stmt = stmt.where(AuditLog.created_at >= since)
    if until:
        stmt = stmt.where(AuditLog.created_at <= until)

    total = await db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = await db.scalars(stmt.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset))
    return AuditPage(items=[r for r in rows], total=total, limit=limit, offset=offset)  # type: ignore[arg-type]
