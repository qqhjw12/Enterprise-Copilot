"""首次启动的引导数据：租户 + 三个角色差异明显的演示账号 + 模型 Catalog。

三个账号是刻意设计的，用来演示"同一问题、不同权限、不同结果"：
  admin@corp.local   —— 平台管理员（全权限，可管项目/模型/审计）
  zhangwei@corp.local —— 普通员工 E1024（看不到薪酬机密文档）
  lijing@corp.local  —— HRBP E2048（可见机密薪酬文档）

幂等：重复启动不会重复插入。
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.scopes import scopes_for_roles
from common.security import hash_password
from platform_api.config import get_settings
from platform_api.models import ModelCatalogEntry, Tenant, User

logger = logging.getLogger(__name__)

DEMO_PASSWORD = "demo12345"

DEMO_USERS = [
    {
        "email": "zhangwei@corp.local",
        "display_name": "张伟",
        "employee_no": "E1024",
        "roles": ["employee"],
    },
    {
        "email": "lijing@corp.local",
        "display_name": "李静",
        "employee_no": "E2048",
        "roles": ["hr"],
        "extra_scopes": ["rag:confidential"],
    },
]

DEFAULT_MODELS = [
    {
        "model_key": "default-chat",
        "display_name": "默认对话模型",
        "model_name": "deepseek-chat",
        "context_window": 64_000,
        "price_in_per_1k": 0.00027,
        "price_out_per_1k": 0.0011,
        "capabilities": ["tool_calling", "json_mode"],
        "is_default": True,
        "is_cheap": False,
    },
    {
        "model_key": "cheap-chat",
        "display_name": "低成本降级模型",
        "model_name": "deepseek-chat",
        "context_window": 32_000,
        "price_in_per_1k": 0.00007,
        "price_out_per_1k": 0.00028,
        "capabilities": ["tool_calling"],
        "is_default": False,
        "is_cheap": True,
    },
]


async def bootstrap(db: AsyncSession) -> None:
    settings = get_settings()

    tenant = await db.scalar(select(Tenant).where(Tenant.slug == settings.bootstrap_tenant_slug))
    if tenant is None:
        tenant = Tenant(slug=settings.bootstrap_tenant_slug, name=settings.bootstrap_tenant_name)
        db.add(tenant)
        await db.flush()
        logger.info("bootstrap_tenant_created", extra={"tenant_id": tenant.id})

    # 管理员
    admin = await db.scalar(
        select(User).where(User.tenant_id == tenant.id, User.email == settings.bootstrap_admin_email)
    )
    if admin is None:
        admin = User(
            tenant_id=tenant.id,
            email=settings.bootstrap_admin_email,
            display_name="平台管理员",
            employee_no="E1000",
            password_hash=hash_password(settings.bootstrap_admin_password),
            roles=["admin"],
            scopes=scopes_for_roles(["admin"]),
        )
        db.add(admin)
        logger.info("bootstrap_admin_created", extra={"tenant_id": tenant.id})

    # 演示账号
    for spec in DEMO_USERS:
        email = spec["email"]
        roles = list(spec["roles"])
        scopes = scopes_for_roles(roles)
        for extra in spec.get("extra_scopes", []):
            if extra not in scopes:
                scopes.append(extra)

        existing = await db.scalar(select(User).where(User.tenant_id == tenant.id, User.email == email))
        if existing is not None:
            # 演示账号的作用域跟随角色定义走：角色表改了（例如给 hr 补上 chat:write），
            # 重启后自动对齐，避免"代码改了但库里的老账号还是旧权限"这种排查半天的问题。
            if set(existing.scopes or []) != set(scopes):
                existing.scopes = scopes
                existing.roles = roles
                logger.info("bootstrap_demo_scopes_synced", extra={"email": email, "scopes": scopes})
            continue
        db.add(
            User(
                tenant_id=tenant.id,
                email=email,
                display_name=spec["display_name"],
                employee_no=spec["employee_no"],
                password_hash=hash_password(DEMO_PASSWORD),
                roles=roles,
                scopes=scopes,
            )
        )

    existing_models = {
        m.model_key
        for m in (await db.scalars(select(ModelCatalogEntry).where(ModelCatalogEntry.tenant_id == tenant.id)))
    }
    for spec in DEFAULT_MODELS:
        if spec["model_key"] not in existing_models:
            db.add(ModelCatalogEntry(tenant_id=tenant.id, **spec))

    await db.commit()
