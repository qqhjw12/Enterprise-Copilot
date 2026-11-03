"""执行面配置。

三个 Settings 类分别对应 RUNTIME_ / RAG_ / MCP_ 前缀，
这样 docker-compose 里能按关注点分组注入环境变量，不必拼一个巨型配置对象。
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RUNTIME_", env_file=".env", extra="ignore")

    app_env: str = "dev"
    platform_base_url: str = "http://localhost:8000"
    jwt_secret: str = "dev-secret-change-me-please"

    redis_url: str = "redis://localhost:6379/0"
    checkpoint_backend: str = "sqlite"  # postgres | sqlite | memory
    checkpoint_dsn: str = ""

    # LLM（OpenAI 兼容协议）
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_cheap_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 60.0

    # 预算（USD）
    run_budget_usd: float = 0.05
    price_in_per_1k: float = 0.00027
    price_out_per_1k: float = 0.0011

    # 图参数
    max_iterations: int = 8
    max_consecutive_failures: int = 2
    history_window: int = 12

    cors_origins: str = "http://localhost:8501,http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    vector_backend: str = "memory"  # memory | qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection: str = "copilot_chunks"

    embedding_provider: str = "hashing"  # hashing | sentence-transformers | openai
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_batch_size: int = 32

    chunker: str = "semantic"  # recursive | semantic
    chunk_size: int = 480
    chunk_overlap: int = 80
    semantic_break_percentile: float = 80.0

    rrf_k: int = 60
    dense_top_k: int = 20
    sparse_top_k: int = 20
    # off | multi_query：后者用受控领域词表扩展 BM25 查询（不增加 embedding 调用）
    query_transform: str = "off"
    query_variants: int = 3
    rerank_top_n: int = 5
    reranker: str = "heuristic"  # heuristic | cross-encoder | none
    reranker_model: str = "BAAI/bge-reranker-base"

    persist_dir: str = "./var/rag"
    knowledge_dir: str = "./data/knowledge"
    # 低于此分数的片段视为"没有依据"，直接丢弃 —— 宁可不答，也不要拿噪声硬编。
    # 该阈值与 reranker 的分数量纲绑定：换成 cross-encoder 后需要重新标定（见 docs）
    score_threshold: float = 0.55


class McpSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_", env_file=".env", extra="ignore")

    ops_url: str = ""          # http 传输：http://localhost:8765/mcp
    ops_command: str = ""      # stdio 传输：python
    ops_args: str = "-m,runtime.mcp_layer.server,--transport,stdio"
    tool_timeout_seconds: float = 8.0
    tool_max_retries: int = 2
    breaker_fail_threshold: int = 5
    breaker_reset_seconds: float = 30.0
    allowed_tools: str = ""    # 逗号分隔白名单，空表示不限制

    @property
    def transport(self) -> str:
        return "http" if self.ops_url else "stdio"

    @property
    def allowed_tool_list(self) -> list[str]:
        return [t.strip() for t in self.allowed_tools.split(",") if t.strip()]


@lru_cache
def get_settings() -> RuntimeSettings:
    return RuntimeSettings()


@lru_cache
def get_rag_settings() -> RagSettings:
    return RagSettings()


@lru_cache
def get_mcp_settings() -> McpSettings:
    return McpSettings()
