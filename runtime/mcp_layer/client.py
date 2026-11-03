"""MCP 客户端：工具发现 + 参数注入 + 韧性包装。

两条装配路径：
  McpToolbox   —— 通过 MultiServerMCPClient 连真实 MCP Server（生产/演示默认）
  LocalToolbox —— 直接调用同进程的 Server 函数（单测、以及 MCP 容器未起时的降级）

两者对外接口完全一致，图节点因此不需要知道"工具从哪来"。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from runtime.config import McpSettings, get_mcp_settings
from runtime.mcp_layer.policies import policy_for
from runtime.mcp_layer.resilience import (
    CircuitBreaker,
    ResiliencePolicy,
    ToolCallError,
    call_with_resilience,
)
from runtime.mcp_layer.schemas import (
    CallerContext,
    CreateTicketInput,
    GetEmployeeAssetsInput,
    OpsResult,
    QueryTicketStatusInput,
    SearchDirectoryInput,
    SubmitLeaveRequestInput,
)

logger = logging.getLogger(__name__)

# 工具契约版本：schema 一改就要动它，客户端/前端据此判断是否需要升级
TOOL_CONTRACT_VERSION = "2026-11-03"

# 本地 Schema 是 tool 契约的唯一真相源：模型看到的参数说明、Server 端的校验都用它
TOOL_INPUT_MODELS = {
    "get_employee_assets": GetEmployeeAssetsInput,
    "query_ticket_status": QueryTicketStatusInput,
    "search_directory": SearchDirectoryInput,
    "submit_leave_request": SubmitLeaveRequestInput,
    "create_ticket": CreateTicketInput,
}


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    write: bool
    scope: str
    title: str = ""
    annotations: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    schema_hash: str = ""


def _schema_hash(name: str, input_schema: dict[str, Any], output_schema: dict[str, Any]) -> str:
    """契约指纹：CI 里比对快照，防止有人改了参数却忘了通知调用方。"""
    payload = json.dumps(
        {"name": name, "input": input_schema, "output": output_schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _annotations(write: bool) -> dict[str, Any]:
    """MCP 2025-06-18 的 ToolAnnotations。

    destructiveHint 显式设 False：不设时协议默认是 True，客户端会把它当危险操作；
    写操作虽然会改数据，但都是"新增申请/工单"，不删除任何东西。
    """
    return {
        "readOnlyHint": not write,
        "destructiveHint": False,
        "idempotentHint": True,  # 写操作靠 idempotency_key 去重，重复提交不会产生两张单
        "openWorldHint": False,  # 只访问企业内部系统，不触达开放互联网
    }


@dataclass(slots=True)
class ToolCallOutcome:
    ok: bool
    tool: str
    args: dict[str, Any]
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_event(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "args": self.args,
            "data": self.data,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            **self.meta,
        }


def _strip_injected_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """把 caller 等"运行时注入"字段从给模型看的 schema 里摘掉。

    模型不应该、也没能力决定"我是谁"。
    """
    schema = json.loads(json.dumps(schema))  # 深拷贝
    props = schema.get("properties", {})
    for field_name in ("caller",):
        props.pop(field_name, None)
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r != "caller"]
    schema.pop("$defs", None)
    return schema


def build_tool_specs() -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    output_schema = OpsResult.model_json_schema()
    for name, model in TOOL_INPUT_MODELS.items():
        policy = policy_for(name)
        if policy is None:
            continue
        parameters = _strip_injected_fields(model.model_json_schema())
        specs[name] = ToolSpec(
            name=name,
            description=policy.label + "。" + policy.approval_hint,
            parameters=parameters,
            write=policy.write,
            scope=policy.scope,
            title=policy.label,
            annotations=_annotations(policy.write),
            output_schema=output_schema,
            schema_hash=_schema_hash(name, parameters, output_schema),
        )
    return specs


class _BaseToolbox:
    degraded: bool = False
    transport: str = "unknown"

    def __init__(self, settings: McpSettings | None = None) -> None:
        self.settings = settings or get_mcp_settings()
        self._specs = build_tool_specs()
        self._breakers = {
            name: CircuitBreaker(
                fail_threshold=self.settings.breaker_fail_threshold,
                reset_seconds=self.settings.breaker_reset_seconds,
                name=name,
            )
            for name in self._specs
        }

    async def start(self) -> None:  # 建立连接/预热
        return None

    async def close(self) -> None:
        return None

    def tool_specs(self) -> list[ToolSpec]:
        allowed = self.settings.allowed_tool_list
        return [s for s in self._specs.values() if not allowed or s.name in allowed]

    def breaker_snapshot(self) -> list[dict[str, Any]]:
        return [b.snapshot() for b in self._breakers.values()]

    def _policy(self, spec: ToolSpec) -> ResiliencePolicy:
        return ResiliencePolicy(
            timeout_seconds=self.settings.tool_timeout_seconds,
            max_retries=self.settings.tool_max_retries,
            # 写操作绝不自动重试：超时后重试可能产生重复的请假单/工单
            retryable=not spec.write,
        )

    async def call(self, name: str, args: dict[str, Any], caller: CallerContext) -> ToolCallOutcome:
        spec = self._specs.get(name)
        if spec is None:
            return ToolCallOutcome(
                ok=False,
                tool=name,
                args=args,
                error={"code": "unknown_tool", "message": f"未注册的工具：{name}"},
            )

        # 服务端再做一次 scope 预检：省掉一次无谓的网络往返，也让错误更早出现
        if not caller.has_scope(spec.scope):
            return ToolCallOutcome(
                ok=False,
                tool=name,
                args=args,
                error={
                    "code": "permission_denied",
                    "message": f"当前账号缺少权限 {spec.scope}",
                    "required_scope": spec.scope,
                },
            )

        payload = dict(args)
        payload["caller"] = caller.model_dump()
        if spec.write and not payload.get("idempotency_key"):
            payload["idempotency_key"] = f"{caller.user_id}-{uuid.uuid4().hex[:12]}"

        started = time.perf_counter()
        try:
            raw = await call_with_resilience(
                lambda: self._invoke(name, payload),
                tool=name,
                policy=self._policy(spec),
                breaker=self._breakers[name],
            )
            latency = (time.perf_counter() - started) * 1000
            body = self._normalize(raw)
            if body.get("ok"):
                return ToolCallOutcome(ok=True, tool=name, args=payload, data=body.get("data"), latency_ms=latency)
            error = body.get("error") or {"code": "tool_failed", "message": "工具返回失败"}
            return ToolCallOutcome(ok=False, tool=name, args=payload, error=error, latency_ms=latency)
        except ToolCallError as exc:
            return ToolCallOutcome(
                ok=False,
                tool=name,
                args=payload,
                error=exc.as_dict(),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    async def _invoke(self, name: str, payload: dict[str, Any]) -> Any:
        raise NotImplementedError

    @staticmethod
    def _normalize(raw: Any) -> dict[str, Any]:
        # MCP CallToolResult（新版适配器直接返回对象）：优先 structuredContent，
        # 其次是 content blocks 里的 JSON 文本。包装层可能是 {"result": {...}}，要拆掉。
        structured = getattr(raw, "structuredContent", None)
        if isinstance(structured, dict):
            return _BaseToolbox._unwrap(structured)
        content = getattr(raw, "content", None)
        if isinstance(content, list):
            return _BaseToolbox._normalize(content)
        if isinstance(raw, dict):
            return _BaseToolbox._unwrap(raw)
        if isinstance(raw, str):
            try:
                return _BaseToolbox._unwrap(json.loads(raw))
            except json.JSONDecodeError:
                return {"ok": True, "data": {"content": raw}}
        if isinstance(raw, list):
            # MCP 适配器返回 content blocks：[{"type": "text", "text": "<json>"}]
            for block in raw:
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if not text:
                    continue
                try:
                    return _BaseToolbox._unwrap(json.loads(text))
                except json.JSONDecodeError:
                    return {"ok": True, "data": {"content": text[:2000]}}
            return {"ok": False, "error": {"code": "empty_result", "message": "工具返回空内容"}}
        if isinstance(raw, tuple):  # (content_blocks, artifact)
            return _BaseToolbox._normalize(list(raw))
        # 兜底：别把对象 repr 直接塞给模型看，只留可读片段
        return {"ok": True, "data": {"content": str(raw)[:2000]}}

    @staticmethod
    def _unwrap(body: dict[str, Any]) -> dict[str, Any]:
        """把 FastMCP 的 {"result": {...}} 包装拆成裸结果。"""
        if set(body) == {"result"} and isinstance(body["result"], dict):
            return body["result"]
        return body


class McpToolbox(_BaseToolbox):
    """通过 MCP 协议调用工具（streamable-http 或 stdio）。

    为什么要自己持一个长连接会话，而不用 `tool.ainvoke()`：
    适配器每次 ainvoke 都会新建一个 ClientSession —— 对 stdio 传输来说就是
    **每次工具调用都重新拉起一个 Server 进程**。实测每次 ~1.4s，全部耗在
    Python 解释器启动上，纯粹是白给。这里持一条会话复用，单次调用降到毫秒级。
    """

    transport = "mcp"

    def __init__(self, settings: McpSettings | None = None) -> None:
        super().__init__(settings)
        self._client = None
        self._remote_tools: dict[str, Any] = {}
        self._server_name = "ops"
        self._session_cm: Any = None
        self._session: Any = None
        self._session_lock = asyncio.Lock()

    def _connection(self) -> dict[str, dict[str, Any]]:
        s = self.settings
        if s.ops_url:
            return {"ops": {"transport": "streamable_http", "url": s.ops_url, "timeout": s.tool_timeout_seconds}}
        args = [a for a in s.ops_args.split(",") if a]
        return {"ops": {"transport": "stdio", "command": s.ops_command or "python", "args": args}}

    async def start(self) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        self._client = MultiServerMCPClient(self._connection())
        try:
            tools = await self._client.get_tools()
        except Exception as exc:  # noqa: BLE001
            self.degraded = True
            logger.error("mcp_discovery_failed", extra={"error": str(exc)})
            raise
        self._remote_tools = {t.name: t for t in tools}
        missing = [s.name for s in self._specs.values() if s.name not in self._remote_tools]
        if missing:
            logger.warning("mcp_tools_missing", extra={"missing": missing})
        await self._ensure_session()
        logger.info("mcp_tools_loaded", extra={"tools": sorted(self._remote_tools)})

    async def _ensure_session(self) -> Any:
        """惰性建立并复用长连接；断了就重建（Server 重启/空闲超时）。"""
        if self._session is not None:
            return self._session
        async with self._session_lock:
            if self._session is not None:
                return self._session
            self._session_cm = self._client.session(self._server_name)
            self._session = await self._session_cm.__aenter__()
            logger.info("mcp_session_open", extra={"server": self._server_name})
            return self._session

    async def _drop_session(self) -> None:
        cm, self._session_cm, self._session = self._session_cm, None, None
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.debug("mcp_session_close_failed", exc_info=True)

    async def _invoke(self, name: str, payload: dict[str, Any]) -> Any:
        if name not in self._remote_tools:
            raise ToolCallError("tool_unavailable", f"远端未暴露工具 {name}", tool=name)
        # 统一按 {"payload": {...}} 传参：与 Server 端函数签名 `def tool(payload: Model)` 对应
        try:
            session = await self._ensure_session()
            return await session.call_tool(name, {"payload": payload})
        except Exception:
            # 会话级失败（管道断了）不算业务失败：重建一次再试，
            # 真正的工具错误由 Server 端以结构化结果返回，不会走到这里。
            await self._drop_session()
            session = await self._ensure_session()
            return await session.call_tool(name, {"payload": payload})

    async def close(self) -> None:
        await self._drop_session()


class LocalToolbox(_BaseToolbox):
    """同进程直调。用于单测与"MCP 容器未启动"的本地开发。"""

    transport = "local"

    def __init__(self, settings: McpSettings | None = None) -> None:
        super().__init__(settings)
        self.degraded = True

    async def _invoke(self, name: str, payload: dict[str, Any]) -> Any:
        from runtime.mcp_layer.server import call_tool

        return await call_tool(name, {"payload": payload})


async def build_toolbox(settings: McpSettings | None = None, *, force_local: bool = False) -> _BaseToolbox:
    """优先连 MCP Server；连不上则降级到本地直调，并记录 degraded 状态。

    注意：降级只发生在"启动时连不上"，运行期失败一律走熔断 + 兜底节点，
    不做"静默换成本地实现"—— 那会让 MCP 的权限边界形同虚设。
    """
    settings = settings or get_mcp_settings()
    if force_local or (not settings.ops_url and not settings.ops_command):
        logger.info("toolbox_local_mode", extra={"reason": "no mcp endpoint configured"})
        return LocalToolbox(settings)

    toolbox = McpToolbox(settings)
    try:
        await toolbox.start()
        toolbox.degraded = False
        return toolbox
    except Exception:  # noqa: BLE001
        logger.error("toolbox_fallback_to_local", extra={"endpoint": settings.ops_url or settings.ops_command})
        return LocalToolbox(settings)
