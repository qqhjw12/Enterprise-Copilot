"""JWT 契约 + 口令哈希。

控制面签发、执行面校验，两边必须共用同一份 claims 约定，
否则"平台治理层与执行层解耦"就会退化成"两边各写一套权限判断"。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

JWT_ISSUER = "enterprise-copilot-platform"
JWT_ALGORITHM = "HS256"
_PBKDF2_ROUNDS = 260_000


class TokenError(Exception):
    """令牌无效/过期。调用方统一映射为 401。"""


@dataclass(slots=True)
class Principal:
    """请求主体：一次调用里所有权限判断的唯一来源。"""

    user_id: str
    tenant_id: str
    email: str = ""
    roles: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    # 业务系统里的工号；工具调用（MCP）认的是它，不是平台主键
    employee_no: str = ""

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "*" in self.scopes or "admin" in self.roles


# ------------------------------------------------------------------ 口令

def hash_password(password: str, *, rounds: int = _PBKDF2_ROUNDS) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), rounds)
    return f"pbkdf2_sha256${rounds}${salt}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds))
    return hmac.compare_digest(digest.hex(), expected)


# ------------------------------------------------------------------ JWT

def _secret(secret: str | None = None) -> str:
    value = secret or os.getenv("PLATFORM_JWT_SECRET") or os.getenv("RUNTIME_JWT_SECRET")
    if not value:
        # 宁可显式报错，也不要静默用一个默认密钥上生产
        raise RuntimeError("JWT secret 未配置：请设置 PLATFORM_JWT_SECRET / RUNTIME_JWT_SECRET")
    return value


def issue_access_token(
    *,
    user_id: str,
    tenant_id: str,
    email: str = "",
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
    employee_no: str = "",
    ttl_minutes: int = 720,
    secret: str | None = None,
) -> str:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": user_id,
        "tid": tenant_id,
        "email": email,
        "emp": employee_no,
        "roles": roles or [],
        "scopes": scopes or [],
        "iss": JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
    }
    return jwt.encode(claims, _secret(secret), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, *, secret: str | None = None) -> Principal:
    try:
        claims = jwt.decode(
            token,
            _secret(secret),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={"require": ["exp", "sub", "tid"]},
        )
    except jwt.PyJWTError as exc:  # 过期、签名不符、算法不符都在这里
        raise TokenError(str(exc)) from exc
    return Principal(
        user_id=claims["sub"],
        tenant_id=claims["tid"],
        email=claims.get("email", ""),
        roles=list(claims.get("roles") or []),
        scopes=list(claims.get("scopes") or []),
        employee_no=str(claims.get("emp") or ""),
    )


def encode_api_suffix(token: str) -> str:
    """给前端拼 Authorization 头用，避免各处重复字符串。"""
    return token


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
