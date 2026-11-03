"""控制面数据模型。

隔离模型刻意做成三段：
  Tenant（租户） → User（用户，带 roles/scopes） → Project（项目，绑定一个 Agent 图）

所有业务表都带 tenant_id 且在查询时强制过滤，而不是"信任上层传参"。
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from platform_api.db import Base


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ten"))
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    users: Mapped[list[User]] = relationship(back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_user_tenant_email"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("usr"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str] = mapped_column(String(128), default="")
    # 业务身份（工号）。平台账号 ↔ 内部系统账号的映射点，工具调用用的就是它
    employee_no: Mapped[str] = mapped_column(String(32), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    roles: Mapped[list[str]] = mapped_column(JSON, default=list)
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tenant: Mapped[Tenant] = relationship(back_populates="users")


class Project(Base):
    """项目 = 一个可对话的 Agent 应用实例（绑定图版本 + 知识库集合 + 模型白名单）。"""

    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("tenant_id", "slug", name="uq_project_tenant_slug"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("prj"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    slug: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    # 图注册名，Runtime 侧据此选择状态图（见 runtime/graph/registry.py）
    graph_key: Mapped[str] = mapped_column(String(64), default="copilot_default")
    collection: Mapped[str] = mapped_column(String(128), default="")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelCatalogEntry(Base):
    """模型 Catalog：租户只能选平台登记过的模型，避免"某个人偷偷接了个境外 API"。"""

    __tablename__ = "model_catalog"
    __table_args__ = (UniqueConstraint("tenant_id", "model_key", name="uq_model_tenant_key"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("mdl"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    model_key: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(128))
    provider: Mapped[str] = mapped_column(String(32), default="openai-compatible")
    base_url: Mapped[str] = mapped_column(String(255), default="")
    model_name: Mapped[str] = mapped_column(String(128))
    context_window: Mapped[int] = mapped_column(Integer, default=32000)
    price_in_per_1k: Mapped[float] = mapped_column(Float, default=0.0)
    price_out_per_1k: Mapped[float] = mapped_column(Float, default=0.0)
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list)  # ["tool_calling","vision"]
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    is_cheap: Mapped[bool] = mapped_column(Boolean, default=False)  # 预算降级目标
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("aud"))
    tenant_id: Mapped[str] = mapped_column(String(40), index=True)
    user_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    resource_type: Mapped[str] = mapped_column(String(64), default="")
    resource_id: Mapped[str] = mapped_column(String(64), default="")
    outcome: Mapped[str] = mapped_column(String(16), default="success")  # success | failure | denied
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ip: Mapped[str] = mapped_column(String(64), default="")
    request_id: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
