"""灌库 CLI。

    # 首次全量
    python -m runtime.rag.ingest --tenant default --path data/knowledge --persist

    # 日常增量：只重建改动过的文档，没变的一律跳过
    python -m runtime.rag.ingest --tenant default --path data/knowledge --persist

    # 先看会动什么，再决定
    python -m runtime.rag.ingest --tenant default --path data/knowledge --dry-run

    # 源目录删了文件，同步清理索引（会删 chunk，谨慎）
    python -m runtime.rag.ingest --tenant default --path data/knowledge --prune --persist

增量判定依赖 index_manifest.json（与向量库同目录）。删掉它等于宣告
"我不知道库里有什么"，下一次运行会全量重建 —— 安全，只是慢。

生产环境的正确姿势是接消息队列做增量同步（见 docs/architecture.md），
CLI 只用于初始化、本地验证与手工修数。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from common.logging import setup_logging
from runtime.config import get_rag_settings
from runtime.rag.pipeline import HybridRAG


async def _run(args: argparse.Namespace) -> int:
    setup_logging("rag-ingest")
    settings = get_rag_settings()
    if args.chunker:
        settings = settings.model_copy(update={"chunker": args.chunker})

    rag = HybridRAG.from_settings(settings)

    # 支持从磁盘恢复已有索引后追加，避免每次全量重建
    resumed = await rag.load_persisted()
    if resumed:
        print(f"已从 {settings.persist_dir} 恢复 {resumed} 个 chunk", file=sys.stderr)

    metadata = json.loads(args.metadata) if args.metadata else {}
    if args.classification:
        metadata["classification"] = args.classification

    target = Path(args.path)
    if not target.exists():
        print(f"路径不存在: {target}", file=sys.stderr)
        return 2

    if args.full:
        # 兼容路径：不做指纹比对，无条件覆盖写入
        from runtime.rag.parsers import load_corpus, parse_file

        docs = (
            [parse_file(target, tenant_id=args.tenant, metadata=metadata)]
            if target.is_file()
            else load_corpus(target, tenant_id=args.tenant, metadata=metadata)
        )
        stats = await rag.index_documents(docs)
        payload = {
            "mode": "full",
            "documents": stats.documents,
            "chunks": stats.chunks,
            "split_seconds": round(stats.split_seconds, 3),
            "embed_seconds": round(stats.embed_seconds, 3),
            "skipped": stats.skipped,
        }
        wrote = stats.chunks
    else:
        if target.is_file():
            from runtime.rag.parsers import parse_file

            report = await rag.sync_documents(
                [parse_file(target, tenant_id=args.tenant, metadata=metadata)],
                prune=False,  # 单文件不该顺手删掉整个知识库
                force=args.force,
                dry_run=args.dry_run,
            )
        else:
            report = await rag.sync_directory(
                target,
                tenant_id=args.tenant,
                metadata=metadata,
                prune=args.prune,
                force=args.force,
                dry_run=args.dry_run,
            )
        payload = {"mode": "dry-run" if args.dry_run else "incremental", "sync": report.as_dict()}
        if report.pending_removal:
            payload["pending_removal_hint"] = (
                f"{len(report.pending_removal)} 份文档已从源中消失但仍在索引里，确认后加 --prune 清理"
            )
        wrote = report.chunks

    if not args.dry_run and args.persist:
        rag.persist()

    payload.update(
        {
            "chunker": rag.chunker.name,
            "embedding": getattr(rag.embedder, "name", "?"),
            "persisted": bool(args.persist and not args.dry_run),
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0
    # 全部 unchanged 也是成功：增量同步的常态就是"什么都不用做"
    return 0 if (wrote or payload.get("sync", {}).get("unchanged")) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise Copilot 知识库灌入（默认增量）")
    parser.add_argument("--tenant", required=True, help="租户 ID（平台侧 tenants.id）")
    parser.add_argument("--path", required=True, help="文件或目录")
    parser.add_argument("--metadata", default="", help='额外元数据 JSON，如 \'{"classification":"internal"}\'')
    parser.add_argument("--classification", default="", help="覆盖密级：public/internal/confidential")
    parser.add_argument("--chunker", default="", choices=["", "recursive", "semantic"])
    parser.add_argument("--persist", action="store_true", help="落盘到 RAG_PERSIST_DIR")
    parser.add_argument("--prune", action="store_true", help="删除已从源目录消失的文档（目录模式）")
    parser.add_argument("--force", action="store_true", help="忽略指纹，全部重建")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写库")
    parser.add_argument("--full", action="store_true", help="退回旧行为：不做比对，无条件覆盖写入")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
