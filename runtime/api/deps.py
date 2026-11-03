"""执行面鉴权：只做"验证"，不做"授权数据管理"。

令牌由控制面签发，执行面用同一个密钥验签。这样：
  - 执行面不需要连平台库，能做到"平台挂了、执行面还能跑存量会话"；
  - 权限变更的最终裁量仍在控制面（改角色后旧 token 会在下次登录/刷新时更新）。
真正的数据级过滤（能不能看某篇文档）在 RAG 的 predicate 里再做一次，双重校验。
"""
from __future__ import annotations

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from common.errors import Forbidden, Unauthorized
from common.security import Principal, TokenError, decode_access_token
from runtime.config import get_settings
from runtime.mcp_layer.schemas import CallerContext

bearer = HTTPBearer(auto_error=False, description="平台签发的 JWT")


async def get_principal(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    if creds is None or not creds.credentials:
        raise Unauthorized("缺少 Bearer 令牌")
    try:
        principal = decode_access_token(creds.credentials, secret=get_settings().jwt_secret)
    except TokenError as exc:
        raise Unauthorized(f"令牌无效：{exc}") from exc
    request.state.principal = principal
    return principal


async def get_caller(principal: Principal = Depends(get_principal)) -> CallerContext:
    """把平台主体转成"业务调用者"：工具层认的是工号，不是平台主键。"""
    return CallerContext(
        user_id=principal.employee_no or principal.user_id,
        tenant_id=principal.tenant_id,
        roles=list(principal.roles),
        scopes=list(principal.scopes),
    )


def require_scope(scope: str):
    async def _dep(caller: CallerContext = Depends(get_caller)) -> CallerContext:
        if not caller.has_scope(scope):
            raise Forbidden("缺少所需权限", details={"missing_scopes": [scope]})
        return caller

    return _dep
