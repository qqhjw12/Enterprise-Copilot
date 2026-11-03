"""日期理解的回归测试。

起点是一个真实反馈：用户说"帮我提交年假从2026年11.3到2026.11.9"，
系统只回一句"格式 YYYY-MM-DD"；改成"2026-11-3 到 2026-11-9"仍然不行。
根因是正则只认 `\\d{4}-\\d{2}-\\d{2}` 这一种写法。

所以这里的用例不是"测正则能不能匹配"，而是把**人真会怎么说**固定下来：
写法可以千变万化，业务只关心 (start, end) 对不对。
"""
from __future__ import annotations

from datetime import date

import pytest

from runtime.llm.client import OfflineRuleChat
from runtime.nlp.dates import parse_date_range, remove_dates

TODAY = date(2026, 11, 3)
LEAVE_TOOLS = [{"type": "function", "function": {"name": "submit_leave_request"}}]


def parse(text: str, today: date = TODAY):
    return parse_date_range(text, today=today)


# --------------------------------------------------------------- 用户原话


@pytest.mark.parametrize(
    "text",
    [
        "帮我提交年假从2026年11.3到2026.11.9",  # 用户实际输入（中文年月 + 点分隔）
        "帮我提交年假从2026-11-3到2026-11-9",  # 用户第二次尝试（单位数月日）
        "帮我提交年假从 2026-11-03 到 2026-11-09",
        "帮我提交年假从2026/11/3至2026/11/9",
        "年假 11月3日 到 11月9日",  # 完全不带年份
        "11月3日~11月9日",
        "从2026年11月3日到2026年11月9日",
    ],
)
def test_real_world_phrasings_all_parse_to_same_range(text):
    result = parse(text)
    assert result.ok, f"没解析出来：{text} → {result.status} / {result.hint}"
    assert result.start == date(2026, 11, 3)
    assert result.end == date(2026, 11, 9)
    assert result.days == 7


def test_smoke_script_phrasing_still_works():
    """冒烟脚本用的就是这句，改造不能把它弄坏。"""
    result = parse("帮我提交年假申请 2026-11-10 到 2026-11-12")
    assert result.ok
    assert result.start == date(2026, 11, 10)
    assert result.end == date(2026, 11, 12)


def test_missing_dates_asks_with_examples_not_a_format_order():
    """追问要示范"能怎么说"，而不是命令用户记一个格式。"""
    result = parse("帮我提交年假")

    assert result.status == "missing"
    assert "YYYY-MM-DD" not in result.hint
    assert "11月3日" in result.hint  # 给了可照抄的例子
    assert "3 天" in result.hint


def test_partial_range_keeps_known_side_and_asks_only_for_the_other():
    """只说了开始日期时，不要重新问一遍两个日期。"""
    result = parse("从2026-11-03开始休年假")

    assert result.status == "partial"
    assert result.start == date(2026, 11, 3)
    assert "2026-11-03" in result.hint  # 回显已记住的部分
    assert "结束" in result.hint


# --------------------------------------------------------------- 补全规则


def test_year_is_inferred_from_today_when_omitted():
    result = parse("11月3日到11月9日")
    assert result.start.year == 2026
    assert any("未写年份" in note for note in result.assumptions)


def test_range_crossing_new_year_rolls_the_end_forward():
    """"12月25日到1月5日"是跨年，不是"结束早于开始"的非法输入。"""
    result = parse("12月25日到1月5日")

    assert result.ok
    assert result.start == date(2026, 12, 25)
    assert result.end == date(2027, 1, 5)


def test_end_day_without_month_borrows_previous_month():
    """"11月30日到3日"：第二个日期只有"日"，要理解成 12 月 3 日。"""
    result = parse("11月30日到3日")

    assert result.ok
    assert result.start == date(2026, 11, 30)
    assert result.end == date(2026, 12, 3)


def test_explicit_reverse_range_is_invalid_not_silently_swapped():
    """两个日期都写全年月还倒序，说明用户自己搞混了 —— 猜一个等于替他签单。"""
    result = parse("2026-11-09到2026-11-03")

    assert result.status == "invalid"
    assert not result.ok
    assert "早于" in result.hint


def test_impossible_calendar_date_is_rejected():
    result = parse("2月30日到3月2日")
    assert not result.ok


def test_same_day_range_is_valid():
    result = parse("11月3日到11月3日")
    assert result.ok
    assert result.days == 1


def test_duration_expression_infers_end_date():
    """「从明天起请 3 天」——只给一个起点和时长，也应该能落成区间。"""
    result = parse("从明天起请3天")

    assert result.ok
    assert result.start == date(2026, 11, 4)
    assert result.end == date(2026, 11, 6)
    assert any("3 天" in note for note in result.assumptions)


def test_chinese_numeral_duration():
    result = parse("11月3日起请三天")
    assert result.ok
    assert result.end == date(2026, 11, 5)


def test_absolute_range_wins_over_duration_word():
    """既给了区间又提到"3天"时，以明确日期为准。"""
    result = parse("11月3日到11月9日，大概3天")
    assert result.ok
    assert result.end == date(2026, 11, 9)


# --------------------------------------------------------------- 事由清洗


@pytest.mark.parametrize(
    "text",
    [
        "帮我提交年假从2026年11.3到2026.11.9",
        "帮我提交年假从2026-11-3到2026-11-9",
        "年假 11月3日 到 11月9日",
    ],
)
def test_remove_dates_leaves_no_date_residue(text):
    """能解析却剥不干净，审批卡片就会显示"因 2026-11-03 到…请假"这种半成品。"""
    cleaned = remove_dates(text)
    assert "2026" not in cleaned
    assert "11月" not in cleaned
    assert "11-3" not in cleaned


# --------------------------------------------------------------- 规则引擎接入


async def _args(text_or_messages):
    messages = text_or_messages if isinstance(text_or_messages, list) else [{"role": "user", "content": text_or_messages}]
    response = await OfflineRuleChat().complete(messages, tools=LEAVE_TOOLS)
    assert response.tool_calls, f"没有发起工具调用：{response.content!r}"
    return response.tool_calls[0].arguments


async def test_engine_calls_tool_for_the_exact_sentence_that_used_to_fail():
    """这条就是用户报的原句，修之前它只会得到一句"格式 YYYY-MM-DD"。"""
    args = await _args("帮我提交年假从2026年11.3到2026.11.9")

    assert args["start_date"] == "2026-11-03"
    assert args["end_date"] == "2026-11-09"
    assert args["leave_type"] == "annual"


async def test_engine_normalizes_single_digit_month_and_day():
    args = await _args("帮我提交年假从2026-11-3到2026-11-9")

    assert args["start_date"] == "2026-11-03"
    assert args["end_date"] == "2026-11-09"


async def test_engine_reason_has_no_date_residue():
    args = await _args("帮我提交年假从2026-11-3到2026-11-9")
    assert "2026" not in args["reason"]
    assert args["reason"]


async def test_engine_fills_end_date_from_next_turn():
    """多轮补全：上一轮给开始，这一轮只补"到 11月9日"。

    只取最后一条用户消息的话，这一轮凑不齐两个日期，用户会被反复追问同一句。
    """
    messages = [
        {"role": "user", "content": "帮我提交年假从2026-11-3"},
        {"role": "assistant", "content": "已记下开始日期 2026-11-03，还需要结束日期。"},
        {"role": "user", "content": "到11月9日"},
    ]
    args = await _args(messages)

    assert args["start_date"] == "2026-11-03"
    assert args["end_date"] == "2026-11-09"


async def test_engine_does_not_carry_dates_from_an_unrelated_turn():
    """上一条在问资产，不能把那里的工号/日期带进请假申请。"""
    messages = [
        {"role": "user", "content": "帮我查下 E2048 的资产"},
        {"role": "assistant", "content": "E2048 名下有 2 台设备。"},
        {"role": "user", "content": "帮我提交年假 11月3日到11月9日"},
    ]
    args = await _args(messages)

    assert args["employee_id"] == "E1024"
    assert args["start_date"] == "2026-11-03"


async def test_engine_asks_back_when_dates_are_truly_missing():
    response = await OfflineRuleChat().complete([{"role": "user", "content": "帮我提交年假"}], tools=LEAVE_TOOLS)

    assert not response.tool_calls
    assert "11月3日" in response.content


# --------------------------------------------------------------- 路由延续


def test_supplement_turn_keeps_operation_intent():
    """「到 11月9日」单看既没动作也没疑问，会被判成知识问答。

    那样同一次请假就被拆成两半：前半句在等日期，后半句跑去查制度。
    """
    from runtime.graph.nodes import classify_intent

    previous = "帮我提交年假从2026-11-3"
    assert classify_intent(previous) == "operation"
    assert classify_intent("到11月9日", previous=previous) == "operation"


def test_supplement_rule_does_not_swallow_a_real_question():
    """上一轮是操作，不代表这一轮的问题也是操作。"""
    from runtime.graph.nodes import classify_intent

    assert classify_intent("年假有几天？", previous="帮我提交年假") == "knowledge"


def test_supplement_without_operation_history_is_not_operation():
    """延续只在"上一轮确实是操作"时成立；上一轮在问制度，这一轮的日期就不该变操作。"""
    from runtime.graph.nodes import classify_intent

    assert classify_intent("到11月9日") == "knowledge"
    assert classify_intent("到11月9日", previous="年假有几天？") == "knowledge"
