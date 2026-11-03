"""Guardrails：输入 / 输出 / 工具参数三层拦截。

设计参照 openai-agents-python 的 guardrail + tripwire 语义：
  - guardrail 只做判定，不负责"怎么回复"；
  - 判定结果 tripped=True 时由调用方立即终止，不再把请求送给模型；
  - 判定结果里**绝不回传命中的敏感原文**（否则日志本身成了泄露渠道）。

三条原则：
  1. 输入拦截必须发生在"调用模型之前"（阻塞式），否则 token 已经花掉了；
  2. 输出拦截是最后一道闸，用于兜住检索/模型侧的意外泄露（跨租户、系统提示、密钥）；
  3. 工具参数拦截在"执行前"，防止模型把 caller 或密钥塞进业务参数里。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- 规则

# 提示注入：覆盖指令、套取系统提示、进入越狱模式、绕过审批/权限
_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "override_instructions",
        re.compile(
            r"(忽略|无视|忘记|跳过|不要遵守).{0,16}(之前|以上|前面|先前|所有|系统|开发者).{0,10}"
            r"(指令|规则|提示|约束|限制|政策)"
        ),
    ),
    (
        "override_instructions",
        re.compile(r"(ignore|disregard|forget)\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions|rules|prompts)", re.I),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"(输出|显示|打印|复述|重复|泄露|告诉我|贴出).{0,12}"
            r"(系统提示|系统指令|system\s*prompt|初始提示|隐藏指令|开发者消息)"
        ),
    ),
    (
        "jailbreak",
        re.compile(r"(开发者模式|developer\s*mode|无限制模式|越狱|DAN模式|do\s+anything\s+now)", re.I),
    ),
    (
        "bypass_control",
        re.compile(r"(绕过|跳过|关掉|禁用).{0,10}(审批|权限|鉴权|安全策略|guardrail|人工确认)"),
    ),
)

_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{24,}=*", re.I)),
    ("generic_secret", re.compile(r"\b(api[_-]?key|secret|password|passwd)\s*[:=]\s*[^\s\"']{8,}", re.I)),
)

_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3}[\dXx])(?!\d)")
_BANK_CARD_RE = re.compile(r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)")

# 跨租户金丝雀：只存在于 ten_b 的探针文档里。正常回答永远不该出现；
# 一旦出现，说明检索过滤或上下文拼装被绕过了。
CROSS_TENANT_CANARY = "ZQ-OTHER-TENANT"

# 系统提示特征串：组合命中才算泄露，避免把"知识库"这类普通词误判
_SYSTEM_PROMPT_MARKERS = ("【检索上下文】", "只能依据【检索上下文】回答", "AGENT_SYSTEM")

INPUT_BLOCK_MESSAGES = {
    "prompt_injection": "你的请求包含试图覆盖系统指令的内容，已被安全策略拦截。请直接描述你要办理的业务。",
    "secret_in_input": "检测到输入中包含疑似密钥或凭据，已按安全策略拦截。请勿在对话中粘贴密钥，改用平台凭据管理。",
    "input_too_long": "输入内容过长，已被安全策略拦截。请缩短后重试。",
}
OUTPUT_BLOCK_MESSAGE = "回答因包含受保护信息已被安全策略拦截。如需该信息，请通过授权流程申请。"
TOOL_ARG_BLOCK_MESSAGE = "工具调用参数未通过安全校验，已阻止执行。"


@dataclass(slots=True)
class Verdict:
    """一次 guardrail 判定的结果。"""

    stage: str  # input | output | tool
    tripped: bool = False
    text: str = ""
    rules: list[str] = field(default_factory=list)
    redactions: int = 0
    latency_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def as_event(self) -> dict[str, Any]:
        # 只回传规则名与计数，不回传命中的原文
        return {
            "stage": self.stage,
            "action": "block" if self.tripped else ("redact" if self.redactions else "allow"),
            "rules": self.rules,
            "redactions": self.redactions,
            "latency_ms": round(self.latency_ms, 3),
            **self.detail,
        }


def _scan(text: str, rules: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    hits: list[str] = []
    for name, pattern in rules:
        if pattern.search(text):
            if name not in hits:
                hits.append(name)
    return hits


def redact_pii(text: str) -> tuple[str, int]:
    """脱敏而非拦截：手机号/身份证/银行卡不该进入模型上下文，但用户可能只是在陈述事实。"""
    count = 0

    def _phone(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        value = match.group(1)
        return f"{value[:3]}****{value[-4:]}"

    def _id_card(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)}********{match.group(3)}"

    def _bank(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)} **** **** {match.group(2)}"

    text = _PHONE_RE.sub(_phone, text)
    text = _ID_CARD_RE.sub(_id_card, text)
    text = _BANK_CARD_RE.sub(_bank, text)
    return text, count


class GuardrailEngine:
    """无状态判定器；可安全地在多请求间共享。"""

    def __init__(self, *, max_input_chars: int = 8000) -> None:
        self.max_input_chars = max_input_chars
    def check_input(self, text: str) -> Verdict:
        started = time.perf_counter()
        verdict = Verdict(stage="input", text=text)

        if len(text) > self.max_input_chars:
            verdict.tripped = True
            verdict.rules.append("input_too_long")
            verdict.text = INPUT_BLOCK_MESSAGES["input_too_long"]
            verdict.latency_ms = (time.perf_counter() - started) * 1000
            return verdict

        injection = _scan(text, _INJECTION_RULES)
        secrets = _scan(text, _SECRET_RULES)
        if injection:
            verdict.tripped = True
            verdict.rules.extend(injection)
            verdict.text = INPUT_BLOCK_MESSAGES["prompt_injection"]
        elif secrets:
            verdict.tripped = True
            verdict.rules.extend(secrets)
            verdict.text = INPUT_BLOCK_MESSAGES["secret_in_input"]
        else:
            sanitized, count = redact_pii(text)
            verdict.text = sanitized
            verdict.redactions = count
            if count:
                verdict.rules.append("pii_redacted")

        verdict.latency_ms = (time.perf_counter() - started) * 1000
        return verdict

    def check_output(self, text: str, *, tenant_id: str = "") -> Verdict:
        started = time.perf_counter()
        verdict = Verdict(stage="output", text=text)

        if CROSS_TENANT_CANARY in text:
            verdict.tripped = True
            verdict.rules.append("cross_tenant_leak")
            verdict.detail["tenant_id"] = tenant_id
            verdict.text = OUTPUT_BLOCK_MESSAGE
        elif _scan(text, _SECRET_RULES):
            verdict.tripped = True
            verdict.rules.append("secret_leak")
            verdict.text = OUTPUT_BLOCK_MESSAGE
        elif sum(1 for marker in _SYSTEM_PROMPT_MARKERS if marker in text) >= 2:
            verdict.tripped = True
            verdict.rules.append("system_prompt_leak")
            verdict.text = OUTPUT_BLOCK_MESSAGE
        else:
            sanitized, count = redact_pii(text)
            verdict.text = sanitized
            verdict.redactions = count
            if count:
                verdict.rules.append("pii_redacted")

        verdict.latency_ms = (time.perf_counter() - started) * 1000
        return verdict

    def check_tool_args(self, tool: str, args: dict[str, Any]) -> Verdict:
        """执行前校验：caller 由运行时注入，模型无权自带；参数里不能夹带密钥。"""
        started = time.perf_counter()
        verdict = Verdict(stage="tool", detail={"tool": tool})

        if "caller" in args:
            verdict.tripped = True
            verdict.rules.append("caller_injection")
            verdict.text = TOOL_ARG_BLOCK_MESSAGE
        else:
            serialized = repr(args)
            secrets = _scan(serialized, _SECRET_RULES)
            if secrets:
                verdict.tripped = True
                verdict.rules.extend(secrets)
                verdict.text = TOOL_ARG_BLOCK_MESSAGE

        verdict.latency_ms = (time.perf_counter() - started) * 1000
        return verdict


# 流式输出里可能"跨 token"泄露：密钥、金丝雀、系统提示原文。
# 这些前缀一旦出现就先扣住，等整体校验通过再放行。
_RISK_PREFIXES = ("ZQ-", "sk-", "AKIA", "-----BEGIN", "【检索上下文】", "只能依据", "AGENT_SYSTEM")


class StreamGuard:
    """逐 token 过闸：宁可晚 8 个字符，也不把半截密钥发出去。

    hold_chars 的作用：正则可能跨 token（"sk-" 与后 16 位分属两个 delta），
    所以尾部几个字符先不释放；风险前缀出现时，从该位置起全部扣住。

    局限（必须写清楚）：这里只拦"阻塞级"泄露（密钥/金丝雀/系统提示）。
    手机号等 PII 的脱敏作用在最终 answer 上，不保证逐 token 脱敏 ——
    要逐 token 脱敏得做流式 DFA，成本和收益不成比例。
    """

    def __init__(self, engine: GuardrailEngine | None = None, *, hold_chars: int = 8) -> None:
        self.engine = engine or get_guardrails()
        self.hold_chars = max(1, hold_chars)
        self.buffer = ""
        self.emitted = 0
        self.tripped: Verdict | None = None

    def feed(self, delta: str) -> tuple[str, Verdict | None]:
        """返回 (可以安全下发的增量, 命中时的判定)。"""
        if self.tripped is not None or not delta:
            return "", self.tripped

        self.buffer += delta
        verdict = self.engine.check_output(self.buffer)
        if verdict.tripped:
            self.tripped = verdict
            return "", verdict

        # 风险前缀之后的内容一律扣住（哪怕还没凑成完整密钥）
        risk_at = len(self.buffer)
        for prefix in _RISK_PREFIXES:
            idx = self.buffer.find(prefix, self.emitted)
            if idx >= 0:
                risk_at = min(risk_at, idx)

        safe_upto = min(risk_at, max(self.emitted, len(self.buffer) - self.hold_chars))
        if safe_upto <= self.emitted:
            return "", None

        chunk = self.buffer[self.emitted : safe_upto]
        self.emitted = safe_upto
        return chunk, None

    def finalize(self) -> tuple[str, Verdict | None]:
        """流结束时释放尾部；若整体校验不过，尾部直接丢弃。"""
        if self.tripped is not None:
            return "", self.tripped

        verdict = self.engine.check_output(self.buffer)
        if verdict.tripped:
            self.tripped = verdict
            return "", verdict

        rest = self.buffer[self.emitted :]
        self.emitted = len(self.buffer)
        return rest, None


_engine = GuardrailEngine()


def get_guardrails() -> GuardrailEngine:
    return _engine
