"""Enterprise Copilot 工作台（Streamlit）。

四个必须存在的交互：
  1. 流式对话 + Markdown 渲染
  2. MCP 工具调用卡片（工具名 / 参数 / 返回）
  3. RAG 引用来源（文档片段 + 相似度分数）
  4. HITL 审批弹窗（批准 / 修改 / 拒绝）

Streamlit 的限制（每次交互整页重跑）决定了状态管理方式：
所有会话状态放 st.session_state，SSE 在单次渲染里同步消费完。
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

RUNTIME_BASE = os.getenv("COPILOT_API_BASE", "http://localhost:8001").rstrip("/")
PLATFORM_BASE = os.getenv("COPILOT_PLATFORM_BASE", "http://localhost:8000").rstrip("/")

DEMO_ACCOUNTS = [
    ("张伟 · 普通员工（工号 E1024）", "zhangwei@corp.local", "demo12345"),
    ("李静 · HRBP（可见机密文档）", "lijing@corp.local", "demo12345"),
    ("管理员 · 全权限", "admin@corp.local", "Copilot@2026"),
]

st.set_page_config(page_title="Enterprise Copilot", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------- 状态

def init_state() -> None:
    st.session_state.setdefault("token", os.getenv("COPILOT_TOKEN", ""))
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("thread_id", "")
    st.session_state.setdefault("history", [])          # [{role, content, citations, tools}]
    st.session_state.setdefault("pending_approval", None)
    st.session_state.setdefault("last_usage", {})
    st.session_state.setdefault("events_log", [])


init_state()


# ---------------------------------------------------------------- 网络

def login(email: str, password: str, tenant: str = "default") -> dict[str, Any]:
    resp = httpx.post(
        f"{PLATFORM_BASE}/v1/auth/login",
        json={"email": email, "password": password, "tenant": tenant},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def stream_sse(url: str, payload: dict[str, Any]):
    """逐行解析 SSE，产出 (event, data)。"""
    headers = {"Authorization": f"Bearer {st.session_state.token}", "Accept": "text/event-stream"}
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=180) as resp:
        if resp.status_code >= 400:
            detail = resp.read().decode(errors="ignore")[:400]
            raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
        event = "message"
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                raw = line[5:].strip()
                try:
                    yield event, json.loads(raw)
                except json.JSONDecodeError:
                    yield event, {"raw": raw}


# ---------------------------------------------------------------- 渲染组件

def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"📚 引用来源（{len(citations)} 段）", expanded=False):
        for idx, cite in enumerate(citations, start=1):
            score = float(cite.get("score") or 0)
            st.markdown(
                f"**[{idx}] {cite.get('title', '未命名')}**  ·  `{cite.get('section', '-')}`  "
                f"·  相似度 **{score:.3f}**"
            )
            st.progress(min(1.0, max(0.0, score)), text=None)
            st.caption(cite.get("snippet", "")[:260])
            scores = cite.get("scores") or {}
            if scores:
                st.code(
                    "  ".join(f"{k}={v}" for k, v in scores.items()), language="text"
                )


def render_tool_card(call: dict[str, Any], result: dict[str, Any] | None) -> None:
    tool = call.get("tool", "unknown")
    badge = "🔐 写操作" if call.get("requires_approval") else "🔍 只读"
    with st.expander(f"🧰 MCP 工具 · `{tool}`  {badge}", expanded=bool(result and not result.get("ok"))):
        st.markdown("**参数**")
        st.json(call.get("args", {}), expanded=False)
        if result is None:
            st.caption("等待审批…" if call.get("requires_approval") else "执行中…")
            return
        st.markdown(f"**结果** · {result.get('latency_ms', 0):.1f} ms")
        if result.get("ok"):
            st.json(result.get("data"), expanded=False)
        else:
            st.error(json.dumps(result.get("error"), ensure_ascii=False))


def render_history() -> None:
    for item in st.session_state.history:
        with st.chat_message(item["role"], avatar="🧑" if item["role"] == "user" else "🤖"):
            st.markdown(item.get("content") or "（空）")
            for call, result in item.get("tools", []):
                render_tool_card(call, result)
            render_citations(item.get("citations") or [])


# ---------------------------------------------------------------- 主流程

def consume_stream(url: str, payload: dict[str, Any]) -> None:
    """消费一次运行的事件流，边到边渲染。"""
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    approval = None
    usage: dict[str, Any] = {}

    placeholder = st.empty()
    status_line = st.status("运行中…", expanded=False)

    try:
        for event, data in stream_sse(url, payload):
            st.session_state.events_log.append({"event": event, "data": data})
            if event == "start":
                st.session_state.thread_id = data.get("thread_id", st.session_state.thread_id)
                status_line.update(label=f"运行中…（模型：{'离线规则引擎' if data.get('offline_llm') else 'LLM'}）")
            elif event == "route":
                status_line.update(label=f"意图路由：{data.get('intent')} ")
            elif event == "tool_call":
                tool_calls.extend(data.get("calls", []))
            elif event == "tool_result":
                tool_results.append(data)
            elif event == "citations":
                citations = data.get("citations", [])
            elif event == "token":
                answer_parts.append(data.get("delta", ""))
                placeholder.markdown("".join(answer_parts))
            elif event == "interrupt":
                approval = data.get("approval")
                st.session_state.pending_approval = data
            elif event == "fallback":
                status_line.update(label=f"已降级：{data.get('reason')}", state="error")
            elif event == "guardrail":
                rules = ", ".join(data.get("rules") or [])
                if data.get("action") == "block":
                    st.error(f"安全策略拦截（{data.get('stage')}）：{rules}")
                else:
                    st.warning(f"敏感信息已脱敏（{data.get('stage')}）：{rules}")
            elif event == "done":
                usage = data.get("usage", {})
                answer_parts = [data.get("answer") or "".join(answer_parts)]
                st.session_state.pending_approval = None if not data.get("awaiting_approval") else st.session_state.pending_approval
            elif event == "error":
                st.error(f"{data.get('code')}: {data.get('message')}")
    except Exception as exc:  # noqa: BLE001
        status_line.update(label="请求失败", state="error")
        st.error(str(exc))
        return
    finally:
        status_line.update(state="complete")

    answer = "".join(answer_parts).strip()
    if approval:
        answer = answer or "⏸️ 已暂停：以下操作需要你确认。"

    placeholder.markdown(answer or "（无输出）")

    paired = list(zip(tool_calls, tool_results + [None] * (len(tool_calls) - len(tool_results)), strict=False))
    for call, result in paired:
        render_tool_card(call, result)
    render_citations(citations)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "tools": paired,
        }
    )
    st.session_state.last_usage = usage
    if approval:
        st.rerun()


def render_approval_panel() -> None:
    pending = st.session_state.pending_approval
    if not pending:
        return
    approval = pending.get("approval") or {}
    st.warning("⏸️ 图执行已中断：以下写操作等待人工审批")
    calls = approval.get("calls") or []
    for call in calls:
        st.markdown(f"**{call.get('tool')}**　参数：")
        st.json(call.get("args", {}), expanded=True)

    edited = st.text_area(
        "如需修改参数，请填写 JSON（留空表示按原参数执行）",
        value="",
        height=100,
        placeholder='{"start_date": "2026-11-15"}',
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ 批准执行", type="primary", use_container_width=True):
            payload: dict[str, Any] = {"action": "approve", "calls": []}
            if edited.strip():
                patch = json.loads(edited)
                payload = {"action": "edit", "calls": [{"id": calls[0]["id"], "args": patch}]}
                payload["action"] = "approve"  # 修改后仍视为批准（edit 语义在服务端合并参数）
            _resume(payload)
    with col2:
        if st.button("❌ 拒绝", use_container_width=True):
            _resume({"action": "reject"})
    with col3:
        st.caption("审批记录会写入平台审计日志；参数修改会体现在审批快照里。")


def _resume(payload: dict[str, Any]) -> None:
    thread_id = st.session_state.thread_id
    st.session_state.pending_approval = None
    st.session_state.history.append(
        {"role": "user", "content": f"（审批操作：{payload['action']}）", "citations": [], "tools": []}
    )
    with st.chat_message("assistant", avatar="🤖"):
        consume_stream(f"{RUNTIME_BASE}/v1/threads/{thread_id}/resume", payload)
    st.rerun()


# ---------------------------------------------------------------- 侧边栏

with st.sidebar:
    st.title("🤖 Enterprise Copilot")
    st.caption(f"Runtime `{RUNTIME_BASE}`　·　Platform `{PLATFORM_BASE}`")

    if not st.session_state.token:
        st.subheader("登录")
        choice = st.selectbox("演示账号", [a[0] for a in DEMO_ACCOUNTS], index=0)
        email, password = DEMO_ACCOUNTS[[a[0] for a in DEMO_ACCOUNTS].index(choice)][1:]
        with st.form("login"):
            email_input = st.text_input("邮箱", value=email)
            password_input = st.text_input("密码", value=password, type="password")
            tenant_input = st.text_input("租户", value="default")
            submitted = st.form_submit_button("登录", use_container_width=True)
        if submitted:
            try:
                result = login(email_input, password_input, tenant_input)
                st.session_state.token = result["access_token"]
                st.session_state.user = result["user"]
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"登录失败：{exc}")
    else:
        user = st.session_state.user or {}
        st.success(f"{user.get('display_name', user.get('email', '已登录'))}")
        st.caption(f"角色：{', '.join(user.get('roles', []))}　工号：{user.get('employee_no', '-')}")
        if st.button("退出登录", use_container_width=True):
            for key in ("token", "user", "history", "thread_id", "pending_approval"):
                st.session_state[key] = "" if isinstance(st.session_state[key], str) else None
            st.session_state.history = []
            st.session_state.pending_approval = None
            st.rerun()

    st.divider()
    if st.button("🆕 新会话", use_container_width=True, disabled=not st.session_state.token):
        st.session_state.thread_id = ""
        st.session_state.history = []
        st.session_state.pending_approval = None
        st.rerun()

    if st.session_state.last_usage:
        usage = st.session_state.last_usage
        st.divider()
        st.caption("本次运行")
        st.metric("成本 (USD)", f"${usage.get('spent_usd', 0):.5f}", help=f"上限 ${usage.get('limit_usd', 0)}")
        st.progress(min(1.0, float(usage.get("ratio", 0))), text=f"预算占用 {usage.get('ratio', 0) * 100:.1f}%")

    with st.expander("调试事件流"):
        st.json(st.session_state.events_log[-25:], expanded=False)


# ---------------------------------------------------------------- 主区域

st.title("企业级 AI Agent 工作台")
st.caption("知识问答走混合检索（Dense + BM25 → RRF → Rerank）；业务操作走 MCP 工具，写操作需人工审批。")

if not st.session_state.token:
    st.info("请在左侧登录后开始对话。默认密码见 `.env.example` 或 README。")
    st.stop()

render_history()
render_approval_panel()

question = st.chat_input("问点什么，或者让我帮你办件事（例如：帮我提交年假申请 2026-11-10 到 2026-11-12）")
if question:
    st.session_state.history.append({"role": "user", "content": question, "citations": [], "tools": []})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🤖"):
        consume_stream(
            f"{RUNTIME_BASE}/v1/chat/stream",
            {"message": question, "thread_id": st.session_state.thread_id},
        )
    st.rerun()
