"""混合检索 RAG 子包。

对外只暴露三个入口：get_rag / ensure_tenant_indexed / retrieve。
其余模块（bm25、fusion、rerank…）属于内部实现细节，允许被单测直接引用，
但不应该被图节点直接调用 —— 图只与 pipeline 打交道。
"""
from runtime.rag.pipeline import HybridRAG
from runtime.rag.service import ensure_tenant_indexed, get_rag, reset, retrieve
from runtime.rag.types import AccessContext, Chunk, Document, RetrievalTrace, RetrievedContext

__all__ = [
    "AccessContext",
    "Chunk",
    "Document",
    "HybridRAG",
    "RetrievalTrace",
    "RetrievedContext",
    "ensure_tenant_indexed",
    "get_rag",
    "reset",
    "retrieve",
]
