from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ auth

class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    # 不在这里卡长度：口令策略属于注册侧，登录侧卡长度只会泄露"口令格式"信息
    password: str = Field(min_length=1, max_length=128)
    tenant: str = Field(default="default", description="租户 slug")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class UserOut(ORMModel):
    id: str
    tenant_id: str
    email: str
    display_name: str
    employee_no: str = ""
    roles: list[str]
    scopes: list[str]
    is_active: bool


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)
    display_name: str = ""
    employee_no: str = Field(default="", max_length=32)
    roles: list[str] = Field(default_factory=lambda: ["employee"])
    scopes: list[str] | None = None


# ------------------------------------------------------------------ project

class ProjectCreate(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    graph_key: str = "copilot_default"
    collection: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    graph_key: str | None = None
    collection: str | None = None
    config: dict[str, Any] | None = None
    is_active: bool | None = None


class ProjectOut(ORMModel):
    id: str
    tenant_id: str
    slug: str
    name: str
    description: str
    graph_key: str
    collection: str
    config: dict[str, Any]
    is_active: bool
    created_at: datetime


# ------------------------------------------------------------------ model catalog

class ModelCatalogCreate(BaseModel):
    model_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")
    display_name: str
    provider: str = "openai-compatible"
    base_url: str = ""
    model_name: str
    context_window: int = 32000
    price_in_per_1k: float = 0.0
    price_out_per_1k: float = 0.0
    capabilities: list[str] = Field(default_factory=lambda: ["tool_calling"])
    is_default: bool = False
    is_cheap: bool = False
    enabled: bool = True


class ModelCatalogOut(ORMModel):
    id: str
    model_key: str
    display_name: str
    provider: str
    model_name: str
    context_window: int
    price_in_per_1k: float
    price_out_per_1k: float
    capabilities: list[str]
    is_default: bool
    is_cheap: bool
    enabled: bool


# ------------------------------------------------------------------ audit

class AuditLogOut(ORMModel):
    id: str
    tenant_id: str
    user_id: str | None
    action: str
    resource_type: str
    resource_id: str
    outcome: str
    detail: dict[str, Any]
    ip: str
    request_id: str
    created_at: datetime


class AuditPage(BaseModel):
    items: list[AuditLogOut]
    total: int
    limit: int
    offset: int


class TenantOut(ORMModel):
    id: str
    slug: str
    name: str
    is_active: bool


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    checks: dict[str, str]
