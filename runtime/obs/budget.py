"""单次运行的预算守卫。

两条线：
  1. 花费超过 70% → 降级到 cheap 模型（还能继续服务，只是质量略降）；
  2. 花费达到 100% → 抛 BudgetExceeded，图进入兜底节点，接口返回 402。

为什么不直接"超了就连环重试"：成本失控本身就是故障，
它对用户的表现必须是"明确告知"而不是"继续烧钱然后给你个烂答案"。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common.errors import BudgetExceeded


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class BudgetGuard:
    limit_usd: float = 0.05
    price_in_per_1k: float = 0.00027
    price_out_per_1k: float = 0.0011
    spent_usd: float = 0.0
    charges: list[dict] = field(default_factory=list)

    def cost_of(self, usage: Usage) -> float:
        return (usage.prompt_tokens / 1000) * self.price_in_per_1k + (
            usage.completion_tokens / 1000
        ) * self.price_out_per_1k

    def charge(self, usage: Usage, *, model: str = "") -> float:
        cost = self.cost_of(usage)
        self.spent_usd += cost
        self.charges.append(
            {"model": model, "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "cost_usd": round(cost, 6)}
        )
        return cost

    @property
    def ratio(self) -> float:
        return self.spent_usd / self.limit_usd if self.limit_usd > 0 else 0.0

    @property
    def should_downgrade(self) -> bool:
        return self.ratio >= 0.7

    @property
    def exhausted(self) -> bool:
        return self.ratio >= 1.0

    def ensure_available(self) -> None:
        if self.exhausted:
            raise BudgetExceeded(
                "本次会话已超出成本预算，请开启新会话或联系管理员调整额度",
                details={"limit_usd": self.limit_usd, "spent_usd": round(self.spent_usd, 6)},
            )

    def snapshot(self) -> dict:
        return {
            "limit_usd": self.limit_usd,
            "spent_usd": round(self.spent_usd, 6),
            "ratio": round(self.ratio, 4),
            "charges": self.charges,
        }
