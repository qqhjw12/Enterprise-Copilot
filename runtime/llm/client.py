"""LLM 客户端实现。

三种实现共用一个极简协议（`async def complete(messages, tools, temperature)`）：
  OpenAICompatChat  —— 走 /chat/completions，DeepSeek / Qwen / vLLM 通吃
  ScriptedChat      —— 单测用：按预设脚本返回，零 API Key、零网络
  OfflineRuleChat   —— 无 Key 时的演示降级：规则解析意图 + 抽取式作答

最后一个不是"假模型"而是产品决策：没有 Key 的人 clone 下来也要能跑通
"工具调用 → 人工审批 → 结果回填"整条链路，否则这个项目没法被评审。
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from runtime.nlp.dates import DateParse, parse_date_range, remove_dates
from runtime.obs.budget import Usage


class LLMError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


@dataclass(slots=True)
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = "stop"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ChatModel(Protocol):
    name: str

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """粗略估算：中文约 1.5 字/token，英文约 4 字符/token。

    生产环境应换成对应模型的 tokenizer；这里只是为了让预算守卫在没有
    精确 usage 的场景（离线模式）也能工作。
    """
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = max(0, len(text) - cjk)
    return max(1, int(cjk / 1.5 + other / 4))


class OpenAICompatChat:
    name = "openai-compatible"

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> LLMResponse:
        import httpx

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise LLMError("llm_unreachable", f"模型服务不可达：{exc}") from exc

        if resp.status_code == 402:
            raise LLMError("llm_payment_required", "上游模型账户余额不足")
        if resp.status_code == 429:
            raise LLMError("llm_rate_limited", "模型服务限流，请稍后重试")
        if resp.status_code >= 400:
            raise LLMError("llm_error", f"模型服务返回 {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage_data = data.get("usage") or {}

        tool_calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                parsed = {}
            tool_calls.append(
                ToolCall(id=item.get("id") or f"call_{uuid.uuid4().hex[:10]}", name=fn.get("name", ""), arguments=parsed)
            )

        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
                completion_tokens=int(usage_data.get("completion_tokens") or 0),
            ),
            model=data.get("model") or (model or self.model),
            finish_reason=choice.get("finish_reason") or "stop",
        )

    async def astream_text(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ):
        """只对"纯文本最终答案"做流式：带 tool_calls 的流式增量拼接极易出错，
        而工具编排阶段本来也不该让用户盯着逐字输出。"""
        import httpx

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="ignore")[:300]
                    raise LLMError("llm_error", f"模型服务返回 {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                    content = delta.get("content")
                    if content:
                        yield content


class ScriptedChat:
    """单测专用：按顺序吐出预设响应；脚本用尽后固定返回最后一条。"""

    name = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(self, messages, *, tools=None, temperature=0.2, model=None) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            return LLMResponse(content="（脚本已用尽）", model="scripted")
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class OfflineRuleChat:
    """无 API Key 的降级实现：规则驱动，行为可预测。"""

    name = "offline-rules"

    _TICKET_ID = re.compile(r"T-\d{4}-\d{4}")
    _EMP_ID = re.compile(r"E\d{4}")
    _LEAVE_TYPES = {"年假": "annual", "病假": "sick", "事假": "personal", "调休": "compensatory"}

    async def complete(self, messages, *, tools=None, temperature=0.2, model=None) -> LLMResponse:
        prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        tool_names = {self._tool_name(t) for t in (tools or [])}
        usage = Usage(prompt_tokens=estimate_tokens(prompt), completion_tokens=80)

        # 已经拿到工具结果：总结即可，不再重复调用工具（模拟 ReAct 的收敛）
        if any(m.get("role") == "tool" for m in messages):
            return self._summarize_tool_results(messages, usage)

        question = self._last_user_text(messages)

        if "get_employee_assets" in tool_names and any(k in question for k in ("资产", "设备", "电脑", "笔记本")):
            return self._call("get_employee_assets", {"employee_id": self._employee_id(question)}, usage)

        if "query_ticket_status" in tool_names and self._TICKET_ID.search(question):
            return self._call("query_ticket_status", {"ticket_id": self._TICKET_ID.search(question).group()}, usage)

        # 请假要多看一步：用户可能在上一轮说了开始日期、这一轮只补"到 11月9日"。
        # 只取最后一条消息会导致"补信息"这一轮永远凑不齐日期。
        leave_text = self._leave_text(messages)
        wants_leave = "请假" in leave_text or ("假" in leave_text and any(k in leave_text for k in ("申请", "提交", "休")))
        if "submit_leave_request" in tool_names and wants_leave:
            parsed = parse_date_range(leave_text)
            leave_type = next((v for k, v in self._LEAVE_TYPES.items() if k in leave_text), "annual")
            if parsed.ok:
                return self._call(
                    "submit_leave_request",
                    {
                        "employee_id": self._employee_id(leave_text),
                        "leave_type": leave_type,
                        **parsed.as_args(),
                        "reason": self._leave_reason(leave_text),
                    },
                    usage,
                )
            return LLMResponse(content=self._date_followup(parsed), model=self.name, usage=usage)

        if "create_ticket" in tool_names and any(k in question for k in ("报修", "创建工单", "提工单")):
            return self._call(
                "create_ticket",
                {"title": question[:60], "description": question[:200], "priority": "P3"},
                usage,
            )

        return LLMResponse(content=self._extractive_answer(question, messages), model=self.name, usage=usage)

    def _leave_reason(self, question: str) -> str:
        """把用户原话洗成一句像样的请假事由。

        直接把整句塞进 reason 会在审批卡片里显示成"帮我提交年假申请 2026-11-10 到…"，
        看着像没加工过的原始输入。这里用与解析器同一套规则剥掉日期（能解析却剥不干净，
        审批卡片就会留下半截日期），再删掉指令词，抽不出就用默认说明。
        """
        text = remove_dates(question)
        for noise in ("帮我", "请帮我", "我想", "我要", "需要", "提交", "申请", "请假", "年假", "事假", "病假", "调休",
                      "一下", "到", "至", "从", "起", "，", ",", "。", "：", ":"):
            text = text.replace(noise, "")
        text = " ".join(text.split()).strip(" -~")
        if len(text) < 2:
            return "因个人事务请假（AI 预填，可在审批时修改）"
        return text[:60]

    def _leave_text(self, messages: list[dict[str, Any]]) -> str:
        """当前问题优先；日期不够两个时，把紧邻的上一条用户消息拼在前面。

        只回溯一条，且要求上一条也在说"请假" —— 否则"帮我查下 E2048 的资产"
        里那个工号、或者上周那次已提交的请假日期，都会被带进新的一次申请。
        """
        users = [m for m in messages if m.get("role") == "user"]
        if not users:
            return ""
        current = str(users[-1].get("content") or "")
        if len(parse_date_range(current).tokens) >= 2:
            return current
        if len(users) >= 2:
            previous = str(users[-2].get("content") or "")
            if "假" in previous:
                return f"{previous}\n{current}"
        return current

    @staticmethod
    def _date_followup(parsed: DateParse) -> str:
        """追问要说明"缺什么"，并给出几种能说的写法，而不是只丢一个格式要求。"""
        lines = [parsed.hint]
        if parsed.assumptions:
            lines.append("（我按这样理解：" + "；".join(parsed.assumptions) + "）")
        return "\n".join(lines)

    # ------------------------------------------------------------ 内部工具

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str:
        return ((tool.get("function") or {}).get("name")) or ""

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    def _employee_id(self, question: str) -> str:
        found = self._EMP_ID.search(question)
        return found.group() if found else "E1024"

    @staticmethod
    def _call(name: str, args: dict[str, Any], usage: Usage) -> LLMResponse:
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args)],
            usage=usage,
            model="offline-rules",
            finish_reason="tool_calls",
        )

    def _summarize_tool_results(self, messages: list[dict[str, Any]], usage: Usage) -> LLMResponse:
        results = [m for m in messages if m.get("role") == "tool"]
        last = results[-1] if results else {"content": "{}"}
        try:
            payload = json.loads(str(last.get("content") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        data = payload.get("data") or {}
        name = str(last.get("name") or "tool")

        # 失败必须先说清楚为什么失败。早先这里是"工具已返回结果。{}"，
        # 把权限拒绝、参数校验失败全都糊成一句废话，用户根本不知道发生了什么。
        if payload.get("ok") is False:
            err = payload.get("error") or {}
            code = str(err.get("code") or "tool_failed")
            reason = str(err.get("message") or "工具执行失败")
            hint = {
                "permission_denied": "该操作超出当前账号权限，可联系管理员申请对应授权。",
                "employee_not_found": "工号在内部系统里查不到，请确认后重试。",
                "validation_error": "参数不合法，请核对后再提交。",
                "timeout": "内部系统响应超时，已自动重试仍未成功，请稍后再试。",
                "circuit_open": "内部系统连续失败，已暂停调用一段时间。",
            }.get(code, "")
            return LLMResponse(
                content=f"操作未完成（{code}）：{reason}{hint and ' ' + hint}",
                model=self.name,
                usage=usage,
            )

        if name == "get_employee_assets" and data:
            items = "、".join(f"{a['name']}（{a['status']}）" for a in data.get("assets", [])) or "无"
            text = f"{data.get('employee_name', '该员工')}（{data.get('department', '')}）名下资产共 {len(data.get('assets', []))} 件：{items}。"
        elif name == "query_ticket_status" and data:
            breach = "，已超出 SLA" if data.get("sla_breached") else "，SLA 内"
            text = f"工单 {data.get('ticket_id')}「{data.get('title')}」当前状态为 {data.get('status')}，负责人 {data.get('assignee')}{breach}。"
        elif name == "submit_leave_request" and data:
            text = f"请假申请已提交，单号 {data.get('request_id')}，共 {data.get('days')} 天，当前状态 {data.get('status')}，下一审批人：{data.get('next_approver')}。"
        elif name == "create_ticket" and data:
            text = f"工单已创建：{data.get('ticket_id')}，优先级 {data.get('priority')}，SLA 到期 {data.get('sla_due_at')}。"
        elif name == "search_directory" and data:
            entries = "；".join(f"{e['name']}（{e['department']}，{e['email']}）" for e in data.get("entries", []))
            text = f"找到 {len(data.get('entries', []))} 位同事：{entries}。"
        elif data:
            text = f"{name} 已返回结果：" + json.dumps(data, ensure_ascii=False)[:200]
        else:
            text = f"{name} 调用成功，但没有返回可展示的数据。"

        return LLMResponse(content=text, model=self.name, usage=usage)

    def _extractive_answer(self, question: str, messages: list[dict[str, Any]]) -> str:
        """从【检索上下文】里挑最相关的句子，带上 [n] 标注。"""
        context = ""
        for message in messages:
            # 必须限定 user 角色：system prompt 里也写着"依据【检索上下文】"，
            # 早先版本按内容匹配会误把 system 当成上下文，导致永远"未收录"
            if message.get("role") != "user":
                continue
            content = str(message.get("content", ""))
            if "【检索上下文】" in content:
                context = content
                break
        if not context:
            return "（离线模式）没有可用的检索上下文。配置 RUNTIME_LLM_API_KEY 后我会用真实模型作答。"

        blocks = re.split(r"\n+(?=\[\d+\])", context)
        candidates: list[tuple[float, str, int]] = []
        q_terms = set(re.findall(r"[\u4e00-\u9fff]{2}|[A-Za-z0-9]+", question))
        for block in blocks:
            match = re.match(r"\[(\d+)\]", block)
            if not match:
                continue
            idx = int(match.group(1))
            for sentence in re.split(r"[。；\n]", block):
                sentence = sentence.strip()
                if len(sentence) < 8:
                    continue
                overlap = sum(1 for t in q_terms if t in sentence)
                if overlap:
                    candidates.append((overlap / max(1, len(q_terms)), sentence, idx))

        if not candidates:
            return "知识库中未收录该信息，建议联系对应部门确认。"
        candidates.sort(key=lambda x: x[0], reverse=True)
        picked = candidates[:3]
        body = "".join(f"{sentence}[{idx}]" for _, sentence, idx in picked)
        return f"根据知识库（离线抽取模式）：{body}"
