"""MCP 工具层单测：Schema 脱敏、权限边界、幂等、熔断与重试策略。"""
from __future__ import annotations

import asyncio

import pytest

from runtime.mcp_layer import backends
from runtime.mcp_layer.client import LocalToolbox, build_tool_specs
from runtime.mcp_layer.policies import TOOL_POLICIES, is_write_tool
from runtime.mcp_layer.resilience import (
    BreakerState,
    CircuitBreaker,
    ResiliencePolicy,
    ToolCallError,
    call_with_resilience,
)
from runtime.mcp_layer.schemas import CallerContext

# ---------------------------------------------------------------- Schema 契约

def test_tool_schemas_do_not_expose_caller():
    """模型不能看到 caller 字段，否则它能伪造别人的身份。"""
    for spec in build_tool_specs().values():
        props = spec.parameters.get("properties", {})
        assert "caller" not in props, f"{spec.name} 暴露了 caller 字段"
        assert "caller" not in (spec.parameters.get("required") or [])


def test_write_tools_are_marked():
    assert is_write_tool("submit_leave_request") is True
    assert is_write_tool("create_ticket") is True
    assert is_write_tool("get_employee_assets") is False


# ---------------------------------------------------------------- 权限边界

@pytest.mark.asyncio
async def test_tool_call_denied_returns_structured_error(caller):
    weak = CallerContext(user_id="E1024", tenant_id="ten_a", roles=["employee"], scopes=[])
    outcome = await LocalToolbox().call("get_employee_assets", {"employee_id": "E1024"}, weak)
    assert outcome.ok is False
    assert outcome.error["code"] == "permission_denied"


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected(caller):
    outcome = await LocalToolbox().call("delete_everything", {}, caller)
    assert outcome.ok is False
    assert outcome.error["code"] == "unknown_tool"


@pytest.mark.asyncio
async def test_cannot_submit_leave_for_others(caller):
    outcome = await LocalToolbox().call(
        "submit_leave_request",
        {"employee_id": "E2048", "leave_type": "annual", "start_date": "2026-11-10", "end_date": "2026-11-11"},
        caller,
    )
    assert outcome.ok is False
    assert outcome.error["code"] == "permission_denied"


@pytest.mark.asyncio
async def test_cannot_read_others_assets(caller):
    """有 ops:asset:read 不等于能查任何人的资产。

    这是最容易漏的一条横向越权：工具级 scope 校验通过，但 employee_id 由模型填，
    模型完全可以把"我的资产"解析成别人的工号。
    """
    outcome = await LocalToolbox().call("get_employee_assets", {"employee_id": "E2048"}, caller)
    assert outcome.ok is False
    assert outcome.error["code"] == "permission_denied"

    own = await LocalToolbox().call("get_employee_assets", {"employee_id": caller.user_id}, caller)
    assert own.ok is True


@pytest.mark.asyncio
async def test_hr_can_read_others_assets(caller):
    """HR 角色可以查他人资产——权限判断是"角色 + scope"两维，不是单一开关。"""
    hr = CallerContext(
        user_id="E2048",
        tenant_id=caller.tenant_id,
        roles=["hr"],
        scopes=["ops:asset:read"],
    )
    outcome = await LocalToolbox().call("get_employee_assets", {"employee_id": "E1024"}, hr)
    assert outcome.ok is True
    assert outcome.data["employee_id"] == "E1024"


@pytest.mark.asyncio
async def test_tool_not_found_is_business_error_not_exception(caller):
    outcome = await LocalToolbox().call("query_ticket_status", {"ticket_id": "T-9999-9999"}, caller)
    assert outcome.ok is False
    assert outcome.error["code"] == "ticket_not_found"


# ---------------------------------------------------------------- 幂等

@pytest.mark.asyncio
async def test_write_tool_is_idempotent_with_same_key(caller):
    backends.reset()
    first, dedup1 = backends.create_leave_request(
        employee_id="E1024",
        leave_type="annual",
        start_date=__import__("datetime").date(2026, 11, 10),
        end_date=__import__("datetime").date(2026, 11, 12),
        reason="test",
        idempotency_key="key-1",
    )
    second, dedup2 = backends.create_leave_request(
        employee_id="E1024",
        leave_type="annual",
        start_date=__import__("datetime").date(2026, 11, 10),
        end_date=__import__("datetime").date(2026, 11, 12),
        reason="test",
        idempotency_key="key-1",
    )
    assert dedup1 is False and dedup2 is True
    assert first.request_id == second.request_id


# ---------------------------------------------------------------- 熔断

@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_fails_fast():
    breaker = CircuitBreaker(fail_threshold=2, reset_seconds=60, name="demo")

    async def boom():
        raise RuntimeError("依赖挂了")

    policy = ResiliencePolicy(timeout_seconds=1, max_retries=0, retryable=False)
    for _ in range(2):
        with pytest.raises(ToolCallError):
            await call_with_resilience(boom, tool="demo", policy=policy, breaker=breaker)

    assert breaker.state is BreakerState.OPEN
    with pytest.raises(ToolCallError) as exc:
        await call_with_resilience(boom, tool="demo", policy=policy, breaker=breaker)
    assert exc.value.code == "circuit_open"


@pytest.mark.asyncio
async def test_read_tool_retries_and_recovers():
    breaker = CircuitBreaker(fail_threshold=5, reset_seconds=1, name="demo")
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("临时故障")
        return "ok"

    result = await call_with_resilience(
        flaky, tool="demo", policy=ResiliencePolicy(max_retries=2, retryable=True, timeout_seconds=1), breaker=breaker
    )
    assert result == "ok"
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_write_tool_does_not_retry():
    """写操作超时后绝不能自动重试 —— 会造成重复请假/重复工单。"""
    breaker = CircuitBreaker(fail_threshold=5, reset_seconds=1, name="demo")
    attempts = {"n": 0}

    async def slow():
        attempts["n"] += 1
        await asyncio.sleep(0.2)
        return "done"

    with pytest.raises(ToolCallError) as exc:
        await call_with_resilience(
            slow, tool="submit_leave_request", policy=ResiliencePolicy(timeout_seconds=0.05, max_retries=3, retryable=False), breaker=breaker
        )
    assert exc.value.code == "timeout"
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_toolbox_marks_write_calls_non_retryable(caller):
    toolbox = LocalToolbox()
    policy = toolbox._policy(toolbox._specs["submit_leave_request"])
    assert policy.retryable is False
    read_policy = toolbox._policy(toolbox._specs["query_ticket_status"])
    assert read_policy.retryable is True


def test_all_tools_have_policy_and_scope():
    for name, policy in TOOL_POLICIES.items():
        assert policy.scope.startswith("ops:"), name
        assert policy.label, name
