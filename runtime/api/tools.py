"""工具目录接口：给平台后台、前端和 CI 一个稳定的契约视图。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from common.errors import ServiceUnavailable
from common.scopes import SCOPE_CHAT_READ
from runtime.api.deps import require_scope
from runtime.mcp_layer.catalog import build_tool_catalog
from runtime.mcp_layer.schemas import CallerContext

router = APIRouter(prefix="/v1", tags=["tools"])


@router.get("/tools")
async def list_tools(
    request: Request,
    caller: CallerContext = Depends(require_scope(SCOPE_CHAT_READ)),
) -> dict[str, Any]:
    toolbox = getattr(request.app.state, "toolbox", None)
    if toolbox is None:
        raise ServiceUnavailable("工具箱尚未装配完成")

    catalog = build_tool_catalog()
    catalog["runtime"] = {
        "transport": toolbox.transport,
        "degraded": toolbox.degraded,
        "allowed": [spec.name for spec in toolbox.tool_specs()],
        "visible_to": {"user_id": caller.user_id, "roles": list(caller.roles)},
    }
    return catalog
