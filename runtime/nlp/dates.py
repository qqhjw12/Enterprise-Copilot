"""把"人怎么说"翻译成日期，而不是要求人按机器格式说。

原来离线规则引擎里只有一行 `re.compile(r"\\d{4}-\\d{2}-\\d{2}")`，于是
"2026年11.3 到 2026.11.9"、"2026-11-3 到 2026-11-9" 全部落空，用户只能收到
一句硬编码的"格式 YYYY-MM-DD"——这不是理解能力问题，是**只认一种写法**。

这个模块把三件事分开，各自可单独延展：

  1. 抽取（_scan）：从任意文本里捞出"像日期"的片段，不关心上下文；
  2. 补全（_resolve）：把缺年、缺月、顺序颠倒的片段补成合法区间；
  3. 追问（DateParse.hint）：缺什么就说什么，并示范几种写法。

新增一种写法只需要扩 `_PATTERNS`，第 2、3 步不用动。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

ParseStatus = Literal["ok", "partial", "missing", "invalid"]

#: 一句话里最多认几个日期。超过两个通常是把时长或工号误当日期，
#: 也可能用户一口气说了整月安排 —— 两种都该让用户确认，而不是替他猜。
MAX_TOKENS = 4


@dataclass(slots=True)
class _Token:
    """一个"像日期"的片段。year/month 为空表示原文没写，交给 _resolve 补。"""

    start: int
    end: int
    day: int = 0
    month: int | None = None
    year: int | None = None
    relative: int | None = None
    raw: str = ""


# 顺序即优先级：先"年月日齐全"，再"月日"，最后"孤零零的 9 日"。
# 三种都扫一遍再按位置去重，避免 "2026-11-03" 被拆成 "11-03"。
# 每条都带 (?<!\d) 边界：否则 "2026-11-03" 会在 "26-11" 处被误切一段出来。
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 2026-11-03 / 2026-11-3 / 2026/11/3 / 2026.11.3 / 2026年11月3日 / 2026年11.3
    re.compile(r"(?<!\d)(?P<year>\d{4})\s*[年\-/.]\s*(?P<month>\d{1,2})\s*[月\-/.]\s*(?P<day>\d{1,2})\s*[日号]?"),
    # 11月3日 / 11/3 / 11.3 / 11-3（无年）
    re.compile(r"(?<!\d)(?P<month>\d{1,2})\s*[月\-/.]\s*(?P<day>\d{1,2})\s*[日号]?"),
    # 9日 / 9号（无年无月，只有跟在"到/至"后面才有意义）
    re.compile(r"(?<!\d)(?P<day>\d{1,2})\s*[日号]"),
)

# 相对日期：只认这四个。不做"下周三"这类需要日历推断的表达——
# 推断就有猜错的可能，猜错就是一张错单。
_RELATIVE: dict[str, int] = {"今天": 0, "今日": 0, "明天": 1, "明日": 1, "后天": 2, "大后天": 3}
_RELATIVE_RE = re.compile("|".join(_RELATIVE))

#: 时长："3 天" / "三个工作日"。用于"从 X 起请 3 天"。
_DURATION_RE = re.compile(r"(\d{1,2})\s*(?:个)?\s*(?:工作日|天)")
_CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_CN_DURATION_RE = re.compile(r"([一二两三四五六七八九十])\s*(?:个)?\s*(?:工作日|天)")


@dataclass(slots=True)
class DateParse:
    """解析结果。用 status 决定做什么，不要靠 start 是否为空去猜。"""

    status: ParseStatus
    start: date | None = None
    end: date | None = None
    assumptions: list[str] = field(default_factory=list)
    hint: str = ""
    tokens: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.start is not None and self.end is not None

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1 if self.ok else 0

    def as_args(self) -> dict[str, str]:
        if not self.ok:
            return {}
        return {"start_date": self.start.isoformat(), "end_date": self.end.isoformat()}


def _scan(text: str, *, limit: int = MAX_TOKENS) -> list[_Token]:
    """抽出所有日期片段，按出现位置排序，重叠的保留更长的那个。"""
    found: list[_Token] = []

    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            day = int(groups.get("day") or 0)
            if not 1 <= day <= 31:
                continue
            month_raw = groups.get("month")
            month = int(month_raw) if month_raw else None
            if month is not None and not 1 <= month <= 12:
                continue
            year_raw = groups.get("year")
            found.append(
                _Token(
                    start=match.start(),
                    end=match.end(),
                    day=day,
                    month=month,
                    year=int(year_raw) if year_raw else None,
                    raw=match.group().strip(),
                )
            )

    for match in _RELATIVE_RE.finditer(text):
        found.append(
            _Token(
                start=match.start(),
                end=match.end(),
                relative=_RELATIVE[match.group()],
                raw=match.group(),
            )
        )

    found.sort(key=lambda t: (t.start, -(t.end - t.start)))
    deduped: list[_Token] = []
    for token in found:
        if deduped and token.start < deduped[-1].end:
            continue
        deduped.append(token)
    return deduped[:limit]


def _duration_days(text: str) -> int | None:
    if match := _DURATION_RE.search(text):
        return int(match.group(1))
    if match := _CN_DURATION_RE.search(text):
        return _CN_DIGITS.get(match.group(1))
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        # 2 月 30 日这类非法日期：不静默修正成 3 月 2 日，宁可判无效让用户确认
        return None


def _bump_month(base: date, day: int) -> date | None:
    year, month = (base.year + 1, 1) if base.month == 12 else (base.year, base.month + 1)
    return _safe_date(year, month, day)


def _resolve(tokens: list[_Token], *, today: date) -> tuple[list[date | None], list[str]]:
    """把片段补成具体日期。返回 (日期列表, 补全说明)。"""
    resolved: list[date | None] = []
    assumptions: list[str] = []

    for token in tokens:
        if token.relative is not None:
            resolved.append(today + timedelta(days=token.relative))
            continue

        previous = resolved[-1] if resolved else None
        year, month = token.year, token.month

        if year is None and month is None:
            # 只有"9日"：必须借前一个日期的年月，否则不知道是哪个月
            if previous is None:
                resolved.append(None)
                continue
            year, month = previous.year, previous.month
            assumptions.append(f"「{token.raw}」未写月份，按 {month} 月理解")
        elif year is None:
            year = previous.year if previous else today.year
            if previous is None:
                assumptions.append(f"「{token.raw}」未写年份，按 {year} 年理解")
        elif month is None:  # pragma: no cover - 当前正则下不会出现，保留防御
            month = previous.month if previous else 1

        candidate = _safe_date(year, month, token.day)  # type: ignore[arg-type]
        if candidate is None:
            resolved.append(None)
            continue

        # 顺序修正：只在"明显是跨月/跨年"时才调整，不去猜更离谱的倒序
        if previous is not None and candidate < previous:
            if token.year is None and token.month is None:
                bumped = _bump_month(previous, token.day)
                if bumped is not None:
                    candidate = bumped
                    assumptions.append(f"「{token.raw}」按 {candidate.month} 月理解（跨月）")
            elif token.year is None:
                crossed = _safe_date(year + 1, month, token.day)  # type: ignore[arg-type]
                if crossed is not None:
                    candidate = crossed
                    assumptions.append(f"「{token.raw}」按 {year + 1} 年理解（跨年）")

        resolved.append(candidate)

    return resolved, assumptions


def parse_date_range(text: str, *, today: date | None = None) -> DateParse:
    """从一句话里解析请假区间。

    成功路径有两条：两个日期（"11月3日到11月9日"）、一个日期 + 时长（"11月3日起请3天"）。
    """
    today = today or date.today()
    tokens = _scan(text or "")
    raws = [t.raw for t in tokens]

    if not tokens:
        return DateParse(
            status="missing",
            hint=(
                "我需要请假的开始和结束日期。下面几种说法都行：\n"
                "· 「年假 11月3日 到 11月9日」\n"
                "· 「2026-11-03 ~ 2026-11-09」\n"
                "· 「从明天起请 3 天」"
            ),
        )

    dates, assumptions = _resolve(tokens, today=today)

    if len(dates) >= 2 and dates[0] and dates[1]:
        start, end = dates[0], dates[1]
        if end < start:
            return DateParse(
                status="invalid",
                assumptions=assumptions,
                tokens=raws,
                hint=f"结束日期 {end.isoformat()} 早于开始日期 {start.isoformat()}，请确认哪个是开始、哪个是结束。",
            )
        return DateParse(status="ok", start=start, end=end, assumptions=assumptions, tokens=raws)

    if dates and dates[0]:
        start = dates[0]
        days = _duration_days(text or "")
        if days and 1 <= days <= 365:
            end = start + timedelta(days=days - 1)
            assumptions.append(f"按「{days} 天」推算结束日期")
            return DateParse(status="ok", start=start, end=end, assumptions=assumptions, tokens=raws)
        return DateParse(
            status="partial",
            start=start,
            assumptions=assumptions,
            tokens=raws,
            hint=(
                f"已记下开始日期 {start.isoformat()}，还需要结束日期。"
                "可以直接说「到 11月9日」，或者告诉我请几天（例如「请 3 天」）。"
            ),
        )

    return DateParse(
        status="invalid",
        assumptions=assumptions,
        tokens=raws,
        hint="这个日期在日历上不存在（比如 2 月 30 日），请再确认一下。",
    )


def remove_dates(text: str) -> str:
    """剥掉文本里的日期片段，用于生成"事由"这类不该带日期冗余的短语。

    与 parse_date_range 共用同一套抽取规则：否则会出现"能解析但剥不干净"，
    审批卡片里就会出现"因个人事务 2026-11-03 到 2026-11-09 请假"这种半成品。
    """
    tokens = _scan(text or "", limit=99)
    if not tokens:
        return text
    out: list[str] = []
    cursor = 0
    for token in sorted(tokens, key=lambda t: t.start):
        out.append(text[cursor : token.start])
        cursor = max(cursor, token.end)
    out.append(text[cursor:])
    return "".join(out)


def merge_user_texts(texts: list[str]) -> str:
    """把多轮里先后补的日期拼起来，让"先给开始、再补结束"也能成立。

    只做顺序拼接，不做语义判断：回溯到哪一轮由上层的会话状态决定，
    否则上一次请假的日期会被带到新的一次申请里。
    """
    return "\n".join(t for t in texts if t).strip()
