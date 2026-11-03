"""测试全局配置。

硬约束：**不依赖任何外部服务**（无 Postgres / 无 Redis / 无 MCP 容器 / 无 API Key），
`pytest` 必须能在一个刚 clone 下来的机器上直接跑绿。
所以在导入任何业务模块之前先把环境变量钉死。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VAR = ROOT / "var" / "test"
VAR.mkdir(parents=True, exist_ok=True)

os.environ.update(
    {
        "APP_ENV": "test",
        "LOG_LEVEL": "WARNING",
        # 执行面
        "RUNTIME_CHECKPOINT_BACKEND": "memory",
        "RUNTIME_JWT_SECRET": "test-secret-key-at-least-32-bytes-long",
        "RUNTIME_LLM_API_KEY": "",
        "RUNTIME_RUN_BUDGET_USD": "0.05",
        "RUNTIME_MAX_ITERATIONS": "4",
        "RUNTIME_MAX_CONSECUTIVE_FAILURES": "2",
        # MCP：不连远端，走同进程直调
        "MCP_OPS_URL": "",
        "MCP_OPS_COMMAND": "",
        # RAG：确定性 embedding，避免下载模型
        "RAG_EMBEDDING_PROVIDER": "hashing",
        "RAG_EMBEDDING_DIM": "512",
        "RAG_RERANKER": "heuristic",
        "RAG_CHUNKER": "recursive",
        "RAG_VECTOR_BACKEND": "memory",
        "RAG_PERSIST_DIR": str(VAR / "rag"),
        "RAG_KNOWLEDGE_DIR": str(ROOT / "data" / "knowledge"),
        # 控制面
        "PLATFORM_DATABASE_URL": f"sqlite+aiosqlite:///{VAR / 'platform.db'}",
        "PLATFORM_JWT_SECRET": "test-secret-key-at-least-32-bytes-long",
        "PLATFORM_BOOTSTRAP_ADMIN_PASSWORD": "Copilot@2026",
    }
)

# 清掉可能残留的测试库
for leftover in [VAR / "platform.db", VAR / "rag"]:
    if leftover.is_dir():
        shutil.rmtree(leftover, ignore_errors=True)
    elif leftover.exists():
        leftover.unlink()


@pytest.fixture(scope="session")
def knowledge_dir() -> Path:
    return ROOT / "data" / "knowledge"


@pytest.fixture
async def rag(knowledge_dir: Path):
    """每个用例一份干净的索引（hashing embedding 下索引全量语料 <50ms）。"""
    from runtime.rag.pipeline import HybridRAG

    pipeline = HybridRAG.from_settings(_rag_settings())
    await pipeline.index_directory(knowledge_dir, tenant_id="ten_a")
    return pipeline


def _rag_settings():
    from runtime.config import RagSettings

    return RagSettings(
        embedding_provider="hashing",
        embedding_dim=512,
        chunker="recursive",
        chunk_size=420,
        chunk_overlap=60,
        reranker="heuristic",
        vector_backend="memory",
        persist_dir=str(VAR / "rag"),
        knowledge_dir=str(ROOT / "data" / "knowledge"),
    )


@pytest.fixture
def rag_settings():
    return _rag_settings()


@pytest.fixture
def employee_ctx():
    from runtime.rag.types import AccessContext

    return AccessContext(
        tenant_id="ten_a",
        user_id="E1024",
        roles=["employee"],
        scopes=["chat:read", "rag:read", "ops:asset:read", "ops:ticket:read", "ops:leave:write"],
    )


@pytest.fixture
def hr_ctx():
    from runtime.rag.types import AccessContext

    return AccessContext(
        tenant_id="ten_a",
        user_id="E2048",
        roles=["hr"],
        scopes=["chat:read", "rag:read", "rag:confidential"],
    )


@pytest.fixture
def caller():
    from runtime.mcp_layer.schemas import CallerContext

    return CallerContext(
        user_id="E1024",
        tenant_id="ten_a",
        roles=["employee"],
        scopes=["ops:asset:read", "ops:ticket:read", "ops:leave:write", "ops:directory:read"],
    )
