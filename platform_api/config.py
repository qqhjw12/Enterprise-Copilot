"""控制面配置。

一个原则：所有外部依赖都能在本地降级跑起来（SQLite / 无 Redis / 无向量库），
否则"一键部署"就变成"先配三天环境"。
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLATFORM_", env_file=".env", extra="ignore")

    app_env: str = "dev"
    database_url: str = "sqlite+aiosqlite:///./var/platform.db"
    jwt_secret: str = "dev-secret-change-me-please"
    jwt_ttl_minutes: int = 720

    bootstrap_admin_email: str = "admin@corp.local"
    bootstrap_admin_password: str = "Copilot@2026"
    bootstrap_tenant_slug: str = "default"
    bootstrap_tenant_name: str = "示例租户"

    cors_origins: str = "http://localhost:8501,http://localhost:5173"
    sql_echo: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> PlatformSettings:
    return PlatformSettings()
