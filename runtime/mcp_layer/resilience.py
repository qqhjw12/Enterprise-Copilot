"""工具调用韧性：CircuitBreaker → Timeout → Retry。

顺序很关键（参考 DeepIntel 的分层）：
  1. 熔断器最外层：已经确定不健康的依赖，连超时都不用等，直接快速失败；
  2. 超时在中间：单次调用不允许拖垮整个请求（MCP 工具经常连接内部老系统）；
  3. 重试最内层：只对"幂等的只读调用"生效，写操作绝不自动重试。

写操作不重试的理由不是"懒"：submit_leave_request 超时后重试可能产生两条请假单，
在 OA 里就是两份审批流。要解决"到底成没成功"，靠 idempotency_key 查询，而不是重试。
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ToolCallError(Exception):
    """结构化的工具失败，供图节点做决策（不向上抛裸异常）。"""

    def __init__(self, code: str, message: str, *, tool: str = "", detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.tool = tool
        self.detail = detail or {}

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "tool": self.tool, "detail": self.detail}


class CircuitBreaker:
    def __init__(self, *, fail_threshold: int = 5, reset_seconds: float = 30.0, name: str = "default") -> None:
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_seconds
        self.name = name
        self._failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and time.monotonic() - self._opened_at >= self.reset_seconds:
            # 冷却结束，允许一个探针请求通过
            return BreakerState.HALF_OPEN
        return self._state

    async def before_call(self) -> None:
        async with self._lock:
            state = self.state
            if state is BreakerState.OPEN:
                remaining = self.reset_seconds - (time.monotonic() - self._opened_at)
                raise ToolCallError(
                    "circuit_open",
                    f"依赖 {self.name} 熔断中，{max(0, remaining):.0f}s 后重试",
                    tool=self.name,
                )
            if state is BreakerState.HALF_OPEN:
                self._state = BreakerState.HALF_OPEN  # 放行探针

    async def on_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._state = BreakerState.CLOSED

    async def on_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._failures >= self.fail_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()
                logger.error("circuit_opened", extra={"breaker": self.name, "failures": self._failures})

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state.value, "failures": self._failures}


@dataclass(slots=True)
class ResiliencePolicy:
    timeout_seconds: float = 8.0
    max_retries: int = 2
    backoff_base: float = 0.2
    retryable: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


async def call_with_resilience(
    fn: Callable[[], Awaitable[Any]],
    *,
    tool: str,
    policy: ResiliencePolicy,
    breaker: CircuitBreaker,
) -> Any:
    await breaker.before_call()

    attempts = policy.max_retries + 1 if policy.retryable else 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(fn(), timeout=policy.timeout_seconds)
            await breaker.on_success()
            logger.info(
                "tool_call_ok",
                extra={"tool": tool, "attempt": attempt, "latency_ms": round((time.perf_counter() - started) * 1000, 2)},
            )
            return result
        except TimeoutError as exc:
            last_error = ToolCallError("timeout", f"{tool} 超时（>{policy.timeout_seconds}s）", tool=tool)
            logger.warning("tool_call_timeout", extra={"tool": tool, "attempt": attempt})
            await breaker.on_failure()
            _ = exc
        except ToolCallError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = ToolCallError("tool_error", f"{tool} 调用失败：{exc}", tool=tool, detail={"type": type(exc).__name__})
            logger.warning("tool_call_error", extra={"tool": tool, "attempt": attempt, "error": str(exc)})
            await breaker.on_failure()

        if attempt < attempts:
            # 指数退避 + 抖动，避免重试风暴把刚恢复的依赖再打挂
            delay = policy.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
            await asyncio.sleep(delay)

    assert last_error is not None
    if isinstance(last_error, ToolCallError):
        raise last_error
    raise ToolCallError("tool_error", str(last_error), tool=tool)
