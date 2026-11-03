"""LLM 网关：模型选择 + 计费 + 降级。

所有模型调用都必须经过这里，而不是让节点直接持有 client ——
否则预算、降级、审计这三件事会在某个节点里被绕过。
"""
from __future__ import annotations

import logging

from runtime.config import RuntimeSettings, get_settings
from runtime.llm.client import (
    ChatModel,
    LLMResponse,
    OfflineRuleChat,
    OpenAICompatChat,
    estimate_tokens,
)
from runtime.obs.budget import BudgetGuard
from runtime.obs.spans import gen_ai_attrs, span

logger = logging.getLogger(__name__)


class LLMGateway:
    def __init__(
        self,
        client: ChatModel,
        *,
        primary_model: str,
        cheap_model: str,
        budget: BudgetGuard,
    ) -> None:
        self.client = client
        self.primary_model = primary_model
        self.cheap_model = cheap_model
        self.budget = budget
        self.degraded_to_cheap = False
        self.offline = isinstance(client, OfflineRuleChat)

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        self.budget.ensure_available()

        model = self.primary_model
        if self.budget.should_downgrade and self.cheap_model != self.primary_model:
            model = self.cheap_model
            if not self.degraded_to_cheap:
                self.degraded_to_cheap = True
                logger.warning(
                    "llm_downgraded_to_cheap_model",
                    extra={"ratio": self.budget.ratio, "model": model},
                )

        with span("chat", **gen_ai_attrs(operation="chat", model=model)) as llm_span:
            response = await self.client.complete(messages, tools=tools, temperature=temperature, model=model)

            # 离线/脚本客户端不返回真实 usage，这里估算，保证预算守卫始终有效
            if response.usage.total_tokens == 0:
                prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
                response.usage.prompt_tokens = estimate_tokens(prompt_text)
                response.usage.completion_tokens = estimate_tokens(response.content)

            cost = self.budget.charge(response.usage, model=response.model or model)
            llm_span.set(
                **{
                    "gen_ai.response.model": response.model or model,
                    "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": response.usage.completion_tokens,
                    "gen_ai.response.finish_reasons": response.finish_reason,
                    "gen_ai.tool.call.count": len(response.tool_calls),
                    "llm.cost_usd": round(cost, 6),
                }
            )
        logger.info(
            "llm_call",
            extra={
                "model": response.model or model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "cost_usd": round(cost, 6),
                "tool_calls": len(response.tool_calls),
                "offline": self.offline,
            },
        )
        return response

    def chat_stream(self, messages: list[dict], *, temperature: float = 0.2):
        """返回异步迭代器；客户端不支持流式时返回 None，由调用方回退到 chat()。"""
        if not hasattr(self.client, "astream_text"):
            return None
        self.budget.ensure_available()

        model = self.primary_model
        if self.budget.should_downgrade and self.cheap_model != self.primary_model:
            model = self.cheap_model

        async def _gen():
            parts: list[str] = []
            with span("chat", **gen_ai_attrs(operation="chat", model=model)) as llm_span:
                async for delta in self.client.astream_text(messages, model=model, temperature=temperature):
                    parts.append(delta)
                    yield delta
                text = "".join(parts)
                # 流式响应拿不到 usage，用估算值记账，保证预算守卫不会被绕过
                from runtime.obs.budget import Usage as BudgetUsage

                usage = BudgetUsage(
                    prompt_tokens=estimate_tokens("\n".join(str(m.get("content", "")) for m in messages)),
                    completion_tokens=estimate_tokens(text),
                )
                self.budget.charge(usage, model=model)
                llm_span.set(
                    **{
                        "gen_ai.response.model": model,
                        "gen_ai.usage.input_tokens": usage.prompt_tokens,
                        "gen_ai.usage.output_tokens": usage.completion_tokens,
                        "gen_ai.response.finish_reasons": "stop",
                        "llm.stream": True,
                    }
                )
            logger.info(
                "llm_stream_done",
                extra={"model": model, "completion_tokens": usage.completion_tokens, "spent_usd": round(self.budget.spent_usd, 6)},
            )

        return _gen()


def build_gateway(
    settings: RuntimeSettings | None = None,
    budget: BudgetGuard | None = None,
    *,
    spec: object | None = None,
) -> LLMGateway:
    """spec 来自控制面的模型 Catalog；缺省时回落到环境变量配置。"""
    settings = settings or get_settings()

    primary = getattr(spec, "primary_model", None) or settings.llm_model
    cheap = getattr(spec, "cheap_model", None) or settings.llm_cheap_model
    price_in = getattr(spec, "price_in_per_1k", 0.0) or settings.price_in_per_1k
    price_out = getattr(spec, "price_out_per_1k", 0.0) or settings.price_out_per_1k

    budget = budget or BudgetGuard(
        limit_usd=settings.run_budget_usd,
        price_in_per_1k=price_in,
        price_out_per_1k=price_out,
    )

    if settings.llm_api_key:
        client: ChatModel = OpenAICompatChat(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=primary,
            timeout=settings.llm_timeout_seconds,
        )
    else:
        logger.warning(
            "llm_offline_mode",
            extra={"hint": "未配置 RUNTIME_LLM_API_KEY，使用规则引擎降级（仅用于链路演示）"},
        )
        client = OfflineRuleChat()

    return LLMGateway(client, primary_model=primary, cheap_model=cheap, budget=budget)
