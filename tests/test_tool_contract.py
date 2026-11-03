"""MCP 工具契约测试：schema / 注解 / 哈希 一旦变化必须显式更新快照。

更新方式：UPDATE_SNAPSHOTS=1 python -m pytest tests/test_tool_contract.py
评审时看 diff 就知道"哪个工具的参数/返回值变了"，而不是等线上模型调错才发现。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from runtime.mcp_layer.catalog import build_tool_catalog

SNAPSHOT = Path(__file__).parent / "snapshots" / "tool_catalog.json"


def test_tool_catalog_matches_snapshot():
    catalog = build_tool_catalog()

    if os.getenv("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert catalog == expected, "工具契约变了：确认调用方已同步，再 UPDATE_SNAPSHOTS=1 更新快照"


def test_caller_never_leaks_into_model_visible_schema():
    catalog = build_tool_catalog()
    for tool in catalog["tools"]:
        assert "caller" not in json.dumps(tool["input_schema"], ensure_ascii=False)
        assert "caller" not in tool["input_schema"].get("properties", {})


def test_annotations_match_write_policy():
    for tool in build_tool_catalog()["tools"]:
        assert tool["annotations"]["readOnlyHint"] is (not tool["write"])
        assert tool["annotations"]["destructiveHint"] is False
        assert tool["annotations"]["openWorldHint"] is False


def test_schema_hash_is_stable_and_distinct():
    hashes = {tool["name"]: tool["schema_hash"] for tool in build_tool_catalog()["tools"]}

    assert len(set(hashes.values())) == len(hashes)
    assert all(len(value) == 16 for value in hashes.values())
    assert hashes == {tool["name"]: tool["schema_hash"] for tool in build_tool_catalog()["tools"]}
