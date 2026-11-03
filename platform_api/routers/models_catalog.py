from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.security import Principal
from platform_api.db import get_db
from platform_api.deps import get_principal
from platform_api.models import ModelCatalogEntry
from platform_api.schemas import ModelCatalogOut

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", response_model=list[ModelCatalogOut])
async def list_models(
    enabled_only: bool = Query(default=True),
    principal: Principal = Depends(get_principal),
    db: AsyncSession = Depends(get_db),
) -> list[ModelCatalogOut]:
    """租户可见的模型清单。

    执行面在每次运行前会读这个 Catalog 决定用哪个模型、按什么单价记账，
    这样"换模型"是治理动作，不需要改 Runtime 代码。
    """
    stmt = select(ModelCatalogEntry).where(ModelCatalogEntry.tenant_id == principal.tenant_id)
    if enabled_only:
        stmt = stmt.where(ModelCatalogEntry.enabled.is_(True))
    rows = await db.scalars(stmt.order_by(ModelCatalogEntry.is_default.desc(), ModelCatalogEntry.model_key))
    return [ModelCatalogOut.model_validate(m) for m in rows]
