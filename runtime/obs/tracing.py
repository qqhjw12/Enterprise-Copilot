"""链路追踪接线。

LangGraph / LangChain 对 LangSmith 的支持是"读环境变量即生效"，
所以这里只做两件事：把配置写进 env、给每次 run 附上可检索的 metadata。
换成 Langfuse 时，只要在 runtime 启动时注册 callback，本文件其余部分不变。
"""
from __future__ import annotations

import logging
import os

from runtime.config import RuntimeSettings

logger = logging.getLogger(__name__)


def configure_tracing(settings: RuntimeSettings) -> dict[str, str]:
    """幂等：重复调用不会覆盖已存在的 env。"""
    enabled: dict[str, str] = {}
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and os.getenv("LANGCHAIN_API_KEY"):
        enabled["langsmith"] = os.getenv("LANGCHAIN_PROJECT", "enterprise-copilot")
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        enabled["langfuse"] = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if enabled:
        logger.info("tracing_enabled", extra={"backends": list(enabled)})
    else:
        logger.info("tracing_disabled", extra={"hint": "设置 LANGCHAIN_TRACING_V2 / LANGFUSE_* 开启"})
    _ = settings
    return enabled


def run_config(
    *,
    tenant_id: str,
    user_id: str,
    thread_id: str,
    request_id: str = "",
    project: str = "copilot_default",
) -> dict:
    """LangGraph 的 configurable + tracing metadata。

    metadata 里的四个 id 让"一次慢请求"能在 LangSmith 里直接被检索到，
    而不需要先按时间翻找。
    """
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"agent-run:{project}",
        "tags": [f"tenant:{tenant_id}", f"project:{project}"],
        "metadata": {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "request_id": request_id,
        },
    }
