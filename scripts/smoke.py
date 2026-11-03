"""端到端冒烟脚本：把「企业级」的每一条承诺都跑一遍，并打印成一张表。

用法（先起 platform-api:8000 与 runtime-service:8001）：
    python scripts/smoke.py
    python scripts/smoke.py --platform http://127.0.0.1:8000 --runtime http://127.0.0.1:8001

它验证的是"行为"，不是"接口能通"：
  1. 知识问答命中 + 引用分数
  2. 语料里没有的问题 → 明确说未收录，而不是编
  3. 同一问题在不同权限下召回不同文档（机密文档只对 HR 可见）
  4. 只读 MCP 工具调用
  5. 写操作触发 HITL interrupt，批准后才真正落库
  6. 拒绝后不执行
  7. 越权调用返回结构化错误（不是裸异常）
  8. 跨租户探针查不到
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

import httpx

EMPLOYEE = {"email": "zhangwei@corp.local", "password": "demo12345"}
HR = {"email": "lijing@corp.local", "password": "demo12345"}
ADMIN = {"email": "admin@corp.local", "password": "Copilot@2026"}


@dataclass
class Row:
    scene: str
    ok: bool
    detail: str


class Client:
    def __init__(self, platform: str, runtime: str) -> None:
        self.platform = platform.rstrip("/")
        self.runtime = runtime.rstrip("/")
        self.http = httpx.Client(timeout=90.0)

    def token(self, account: dict[str, str]) -> str:
        resp = self.http.post(f"{self.platform}/v1/auth/login", json={**account, "tenant": "default"})
        resp.raise_for_status()
        return resp.json()["access_token"]

    def chat(self, token: str, message: str) -> dict[str, Any]:
        resp = self.http.post(
            f"{self.runtime}/v1/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": message},
        )
        resp.raise_for_status()
        return resp.json()

    def resume(self, token: str, thread_id: str, action: str) -> dict[str, Any]:
        resp = self.http.post(
            f"{self.runtime}/v1/threads/{thread_id}/resume",
            headers={"Authorization": f"Bearer {token}"},
            json={"action": action},
        )
        resp.raise_for_status()
        return _parse_sse(resp.text)

    def close(self) -> None:
        self.http.close()


def _parse_sse(body: str) -> dict[str, Any]:
    """把 SSE 文本解析成 {"final": {...}, "events": [...]}，与 /v1/chat 返回结构对齐。"""
    events: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = ""
        payload: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip() or "{}")
        events.append({"event": name, "data": payload})
        if name in {"done", "interrupt", "error"}:
            final = {"event": name, "data": payload}
    return {"final": final, "events": events}


def run(platform: str, runtime: str) -> list[Row]:
    c = Client(platform, runtime)
    rows: list[Row] = []
    try:
        emp = c.token(EMPLOYEE)
        hr = c.token(HR)

        # 1. 知识问答
        d = c.chat(emp, "年假有几天？")["final"]["data"]
        cites = d.get("citations") or []
        rows.append(
            Row(
                "知识问答（员工）",
                bool(cites) and "未收录" not in (d.get("answer") or ""),
                f"{len(cites)} 条引用，top1={cites[0]['title'] if cites else '-'}({cites[0]['score'] if cites else 0})",
            )
        )

        # 2. 语料外问题
        d = c.chat(emp, "公司有没有太空电梯补贴？")["final"]["data"]
        rows.append(
            Row(
                "语料外问题",
                not (d.get("citations") or []) or "未收录" in (d.get("answer") or ""),
                (d.get("answer") or "")[:46],
            )
        )

        # 3. 权限隔离：同一问题两种身份
        q = "补充医疗保险报销上限是多少"
        d_emp = c.chat(emp, q)["final"]["data"]
        d_hr = c.chat(hr, q)["final"]["data"]
        emp_titles = {x["title"] for x in d_emp.get("citations") or []}
        hr_titles = {x["title"] for x in d_hr.get("citations") or []}
        leaked = any("薪酬" in t for t in emp_titles)
        rows.append(
            Row(
                "权限隔离（机密文档）",
                (not leaked) and any("薪酬" in t for t in hr_titles),
                f"员工可见={sorted(emp_titles) or '无'} / HR 额外可见={sorted(hr_titles - emp_titles) or '无'}",
            )
        )

        # 4. 只读工具
        raw = c.chat(emp, "帮我查一下我名下的资产")
        results = [e["data"] for e in raw["events"] if e["event"] == "tool_result"]
        rows.append(
            Row(
                "只读 MCP 工具",
                bool(results) and results[-1].get("ok") is True,
                f"{results[-1]['tool']} ok={results[-1]['ok']} 耗时={results[-1].get('latency_ms')}ms"
                if results
                else "无工具调用",
            )
        )

        # 5. 写操作：批准
        raw = c.chat(emp, "帮我提交年假申请 2027-05-06 到 2027-05-08")
        f = raw["final"]
        tid = f["data"]["thread_id"]
        approved = c.resume(emp, tid, "approve")["final"]["data"]
        rows.append(
            Row(
                "写操作 HITL · 批准",
                f["event"] == "interrupt" and "已提交" in (approved.get("answer") or ""),
                approved.get("answer", "")[:60],
            )
        )

        # 6. 写操作：拒绝
        raw = c.chat(emp, "帮我提交年假申请 2027-06-01 到 2027-06-02")
        tid = raw["final"]["data"]["thread_id"]
        rejected = c.resume(emp, tid, "reject")["final"]["data"]
        rows.append(
            Row(
                "写操作 HITL · 拒绝",
                "取消" in (rejected.get("answer") or "") and "未在内部系统" in (rejected.get("answer") or ""),
                rejected.get("answer", "")[:60],
            )
        )

        # 7. 越权：员工查他人资产
        raw = c.chat(emp, "帮我查一下 E2048 名下的资产")
        results = [e["data"] for e in raw["events"] if e["event"] == "tool_result"]
        denied = bool(results) and results[-1].get("ok") is False
        code = (results[-1].get("error") or {}).get("code") if results else "-"
        rows.append(
            Row("越权调用（结构化错误）", denied, f"code={code} answer={(raw['final']['data'].get('answer') or '')[:40]}")
        )

        # 8. 跨租户探针
        d = c.chat(emp, "ZQ-OTHER-TENANT-9931")["final"]["data"]
        rows.append(
            Row("跨租户隔离", not (d.get("citations") or []), (d.get("answer") or "")[:46])
        )
    finally:
        c.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise Copilot 端到端冒烟")
    parser.add_argument("--platform", default="http://127.0.0.1:8000")
    parser.add_argument("--runtime", default="http://127.0.0.1:8001")
    args = parser.parse_args(argv)

    rows = run(args.platform, args.runtime)
    width = max(len(r.scene) for r in rows) + 2
    print(f"{'场景'.ljust(width)}结果    细节")
    print("-" * 96)
    for row in rows:
        mark = "PASS" if row.ok else "FAIL"
        print(f"{row.scene.ljust(width)}{mark:<8}{row.detail}")
    print("-" * 96)
    failed = [r for r in rows if not r.ok]
    print(f"合计 {len(rows)} 项，通过 {len(rows) - len(failed)} 项")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
