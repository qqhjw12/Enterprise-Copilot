"""模型 Catalog 客户端。

控制面是"用哪个模型、按什么单价记账"的唯一真相源；执行面只做缓存与兜底。
缓存按租户 60s，避免每次对话都多打一次控制面。

设计取舍：Catalog 不可用时**降级到环境变量配置**而不是报错。
理由：控制面短时抖动不应该让整个对话不可用；但会打 WARN 并标记 degraded，
让运维能在监控里看见"运行在无治理模式下"。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelSpec:
    primary_model: str
    cheap_model: str
    price_in_per_1k: float
    price_out_per_1k: float
    source: str = "platform-catalog"


class ModelCatalogClient:
    def __init__(
        self,
        base_url: str,
        *,
        ttl_seconds: float = 60.0,
        negative_ttl_seconds: float = 30.0,
        timeout: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ttl = ttl_seconds
        self.negative_ttl = negative_ttl_seconds
        self.timeout = timeout
        self._cache: dict[str, tuple[float, ModelSpec]] = {}
        # 控制面不可用时不能每次对话都去等超时：失败结果也要缓存一小段时间
        self._negative_until: dict[str, float] = {}

    async def resolve(self, token: str, tenant_id: str) -> ModelSpec | None:
        cached = self._cache.get(tenant_id)
        if cached and time.monotonic() - cached[0] < self.ttl:
            return cached[1]
        if not self.base_url or not token:
            return None
        if time.monotonic() < self._negative_until.get(tenant_id, 0.0):
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"enabled_only": "true"},
                )
            if resp.status_code != 200:
                logger.warning("model_catalog_http_error", extra={"status": resp.status_code})
                self._mark_unavailable(tenant_id)
                return None
            entries = [m for m in resp.json() if m.get("enabled", True)]
        except httpx.HTTPError as exc:
            logger.warning("model_catalog_unreachable", extra={"error": str(exc)})
            self._mark_unavailable(tenant_id)
            return None

        if not entries:
            return None

        default = next((m for m in entries if m.get("is_default")), entries[0])
        cheap = next((m for m in entries if m.get("is_cheap")), default)
        spec = ModelSpec(
            primary_model=default["model_name"],
            cheap_model=cheap["model_name"],
            price_in_per_1k=float(default.get("price_in_per_1k") or 0.0),
            price_out_per_1k=float(default.get("price_out_per_1k") or 0.0),
        )
        self._cache[tenant_id] = (time.monotonic(), spec)
        logger.info(
            "model_catalog_resolved",
            extra={"tenant_id": tenant_id, "primary": spec.primary_model, "cheap": spec.cheap_model},
        )
        return spec

    def _mark_unavailable(self, tenant_id: str) -> None:
        self._negative_until[tenant_id] = time.monotonic() + self.negative_ttl

    def invalidate(self, tenant_id: str | None = None) -> None:
        if tenant_id is None:
            self._cache.clear()
            self._negative_until.clear()
        else:
            self._cache.pop(tenant_id, None)
            self._negative_until.pop(tenant_id, None)
