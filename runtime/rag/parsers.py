"""文档解析层：PDF / Markdown / HTML → 统一 Document。

统一输出"类 Markdown"文本（标题用 # 表示），这样切分层可以只认一种格式，
而引用卡片又能拿到 section 路径。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from runtime.rag.types import Document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm"}


def _doc_id(path: Path) -> str:
    digest = hashlib.blake2b(str(path).encode(), digest_size=8).hexdigest()
    return f"doc_{path.stem[:24]}_{digest}"


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """极简 front matter：---

    tenant: ten_xxx
    classification: confidential
    allowed_roles: hr,finance
    ---

    只支持 key: value 与逗号分隔列表，够用且不引入 YAML 依赖。
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if len(lines) < 2:
        return {}, text
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text

    meta: dict[str, Any] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key, raw = key.strip(), raw.strip()
        if not key:
            continue
        if "," in raw:
            meta[key] = [item.strip() for item in raw.split(",") if item.strip()]
        elif raw.lower() in {"true", "false"}:
            meta[key] = raw.lower() == "true"
        else:
            meta[key] = raw
    return meta, "\n".join(lines[end + 1 :])


def parse_markdown(path: Path) -> tuple[str, dict[str, Any]]:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8", errors="ignore"))
    return _clean(body), meta


def parse_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    lines: list[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = " ".join(node.get_text(" ", strip=True).split())
        if not text:
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            lines.append(f"{'#' * int(node.name[1])} {text}")
        elif node.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return _clean("\n\n".join(lines))


def parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        # 页码写成二级标题：切分后 section 天然等于"第 N 页"，引用可定位到页
        content = (page.extract_text() or "").strip()
        if content:
            pages.append(f"## 第 {idx} 页\n\n{content}")
    return _clean("\n\n".join(pages))


def parse_file(path: str | Path, *, tenant_id: str = "", metadata: dict[str, Any] | None = None) -> Document:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文档类型: {suffix}")

    if suffix in {".md", ".markdown", ".txt"}:
        text, front = parse_markdown(file_path)
    elif suffix in {".html", ".htm"}:
        text, front = parse_html(file_path), {}
    else:
        text, front = parse_pdf(file_path), {}

    if not text.strip():
        raise ValueError(f"文档内容为空或无法提取文本: {file_path.name}")

    merged = {**front, **(metadata or {})}
    return Document(
        doc_id=_doc_id(file_path),
        title=merged.get("title") or file_path.stem,
        text=text,
        source=str(file_path).replace("\\", "/"),
        tenant_id=str(merged.get("tenant") or tenant_id),
        metadata={"suffix": suffix, **merged},
    )


def load_corpus(root: str | Path, *, tenant_id: str = "", metadata: dict[str, Any] | None = None) -> list[Document]:
    """递归加载目录。单个文件解析失败不影响整体 —— 企业知识库永远有脏文件。"""
    root_path = Path(root)
    docs: list[Document] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            docs.append(parse_file(path, tenant_id=tenant_id, metadata=metadata))
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "document_parse_failed", extra={"file": str(path), "error": str(exc)}
            )
    return docs
