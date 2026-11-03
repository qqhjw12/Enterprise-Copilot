"""端到端验证：请假日期的多种写法与多轮补全。

这个脚本对应一次真实反馈：用户三次表达请假区间，前两次都只收到
"请提供请假的开始与结束日期（格式 YYYY-MM-DD）"。原因不是模型笨，
是规则引擎里只有 `\\d{4}-\\d{2}-\\d{2}` 一条正则。

跑之前需要 8000 / 8001 在线：

    python scripts/smoke.py          # 先确认基础链路正常
    python scripts/verify_dates.py   # 再验证日期理解

退出码 0 表示全部通过。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# 复用冒烟脚本的账号与地址，避免口令/端口在第二个脚本里再维护一份
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke import EMPLOYEE  # noqa: E402

PLATFORM = "http://127.0.0.1:8000"
RUNTIME = "http://127.0.0.1:8001"


@dataclass
class Row:
    scene: str
    ok: bool
    detail: str


class Client:
    def __init__(self) -> None:
        self.http = httpx.Client(timeout=90.0)

    def token(self) -> str:
        resp = self.http.post(f"{PLATFORM}/v1/auth/login", json={**EMPLOYEE, "tenant": "default"})
        resp.raise_for_status()
        return resp.json()["access_token"]

    def chat(self, token: str, message: str, thread_id: str = "") -> dict[str, Any]:
        body: dict[str, Any] = {"message": message}
        if thread_id:
            body["thread_id"] = thread_id
        resp = self.http.post(f"{RUNTIME}/v1/chat", headers={"Authorization": f"Bearer {token}"}, json=body)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self.http.close()


def _call_args(raw: dict[str, Any]) -> dict[str, Any]:
    """从 interrupt 负载里取第一个待审批调用的参数。

    审批内容放在 data.approval 下（data 本身还有 thread_id/usage 等），
    这里两处都试一下，免得协议调整后脚本静默取到空字典。
    """
    final = raw.get("final") or {}
    if final.get("event") != "interrupt":
        return {}
    data = final.get("data") or {}
    for container in (data.get("approval"), data):
        calls = (container or {}).get("calls") or []
        if calls:
            return calls[0].get("args") or {}
    return {}


def _answer(raw: dict[str, Any]) -> str:
    return str(((raw.get("final") or {}).get("data") or {}).get("answer") or "")


def run() -> list[Row]:
    c = Client()
    rows: list[Row] = []
    try:
        token = c.token()

        # 1. 完全没给日期：应该追问，且给出可照抄的写法（而不是一个格式要求）
        raw = c.chat(token, "帮我提交年假")
        answer = _answer(raw)
        rows.append(
            Row(
                "无日期 → 追问示例",
                (raw["final"]["event"] == "done") and "11月3日" in answer and "YYYY-MM-DD" not in answer,
                answer.replace("\n", " ")[:70],
            )
        )

        # 2. 用户原话一：中文年月 + 点分隔
        raw = c.chat(token, "帮我提交年假从2026年11.3到2026.11.9")
        args = _call_args(raw)
        rows.append(
            Row(
                "中文年月点分隔",
                args.get("start_date") == "2026-11-03" and args.get("end_date") == "2026-11-09",
                f"{args.get('start_date')} → {args.get('end_date')}",
            )
        )

        # 3. 用户原话二：单位数月日
        raw = c.chat(token, "帮我提交年假从2026-11-3到2026-11-9")
        args = _call_args(raw)
        rows.append(
            Row(
                "单位数月日",
                args.get("start_date") == "2026-11-03" and args.get("end_date") == "2026-11-09",
                f"{args.get('start_date')} → {args.get('end_date')}",
            )
        )

        # 4. 多轮补全：先给开始，再补"到 11月9日"
        first = c.chat(token, "帮我提交年假从2026-11-3")
        thread_id = ((first.get("final") or {}).get("data") or {}).get("thread_id") or ""
        reminder = _answer(first)
        second = c.chat(token, "到11月9日", thread_id=thread_id)
        args = _call_args(second)
        rows.append(
            Row(
                "多轮补全结束日期",
                ("2026-11-03" in reminder) and args.get("end_date") == "2026-11-09",
                f"追问含已记住的日期；补全后 {args.get('start_date')} → {args.get('end_date')}",
            )
        )

        # 5. 事由不应该残留日期
        rows.append(
            Row(
                "事由无日期残留",
                "2026" not in str(args.get("reason", "")) and bool(args.get("reason")),
                str(args.get("reason"))[:50],
            )
        )
    finally:
        c.close()
    return rows


def main() -> int:
    rows = run()
    width = 22
    print(f"{'场景'.ljust(width)}结果    细节")
    print("-" * 96)
    for row in rows:
        print(f"{row.scene.ljust(width)}{'PASS' if row.ok else 'FAIL'}    {row.detail}")
    print("-" * 96)
    passed = sum(1 for r in rows if r.ok)
    print(f"合计 {len(rows)} 项，通过 {passed} 项")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
