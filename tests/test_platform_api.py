"""控制面接口测试：认证、多租户隔离、审计。"""
from __future__ import annotations

import uuid

import httpx
import pytest

from common.scopes import scopes_for_roles
from common.security import hash_password
from platform_api.main import app


@pytest.fixture
async def client():
    from platform_api.bootstrap import bootstrap
    from platform_api.db import dispose_engine, get_sessionmaker, init_db

    await init_db()
    async with get_sessionmaker()() as session:
        await bootstrap(session)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://platform") as http:
        yield http
    await dispose_engine()


async def login(client: httpx.AsyncClient, email: str = "admin@corp.local", password: str = "Copilot@2026", tenant: str = "default"):
    resp = await client.post("/v1/auth/login", json={"email": email, "password": password, "tenant": tenant})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_health_and_ready(client):
    assert (await client.get("/health")).json()["status"] == "ok"
    ready = await client.get("/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["database"] == "ok"


@pytest.mark.asyncio
async def test_login_success_and_failure(client):
    token = await login(client)
    me = await client.get("/v1/auth/me", headers=auth(token))
    assert me.status_code == 200
    assert me.json()["email"] == "admin@corp.local"
    assert me.json()["employee_no"] == "E1000"

    bad = await client.post(
        "/v1/auth/login", json={"email": "admin@corp.local", "password": "wrong", "tenant": "default"}
    )
    assert bad.status_code == 401
    assert bad.json()["error"]["code"] == "unauthorized"


@pytest.mark.asyncio
async def test_requires_token(client):
    resp = await client.get("/v1/projects")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_project_crud_and_audit(client):
    token = await login(client)
    slug = f"proj-{uuid.uuid4().hex[:6]}"

    created = await client.post("/v1/projects", headers=auth(token), json={"slug": slug, "name": "测试项目"})
    assert created.status_code == 201, created.text
    project = created.json()
    assert project["collection"].startswith("ten_")

    listed = await client.get("/v1/projects", headers=auth(token))
    assert any(p["id"] == project["id"] for p in listed.json())

    audit = await client.get("/v1/audit", headers=auth(token), params={"action": "project.create"})
    assert audit.status_code == 200
    assert any(item["resource_id"] == project["id"] for item in audit.json()["items"])

    login_audit = await client.get("/v1/audit", headers=auth(token), params={"action": "auth.login"})
    assert login_audit.json()["total"] >= 1


@pytest.mark.asyncio
async def test_model_catalog_is_tenant_scoped(client):
    token = await login(client)
    models = await client.get("/v1/models", headers=auth(token))
    assert models.status_code == 200
    keys = {m["model_key"] for m in models.json()}
    assert {"default-chat", "cheap-chat"} <= keys
    assert any(m["is_cheap"] for m in models.json())


@pytest.mark.asyncio
async def test_tenant_isolation_on_projects(client):
    """B 租户拿着自己的 token 访问 A 租户的项目，必须 404（而不是 403，避免泄露存在性）。"""
    from platform_api.db import get_sessionmaker
    from platform_api.models import Tenant, User

    slug = f"other-{uuid.uuid4().hex[:6]}"
    email = f"admin@{slug}.local"
    async with get_sessionmaker()() as session:
        tenant = Tenant(slug=slug, name="另一租户")
        session.add(tenant)
        await session.flush()
        session.add(
            User(
                tenant_id=tenant.id,
                email=email,
                display_name="另一租户管理员",
                employee_no="E9001",
                password_hash=hash_password("Copilot@2026"),
                roles=["admin"],
                scopes=scopes_for_roles(["admin"]),
            )
        )
        await session.commit()

    token_a = await login(client)
    token_b = await login(client, email=email, password="Copilot@2026", tenant=slug)

    created = await client.post(
        "/v1/projects", headers=auth(token_a), json={"slug": f"iso-{uuid.uuid4().hex[:6]}", "name": "A 的项目"}
    )
    project_id = created.json()["id"]

    cross = await client.get(f"/v1/projects/{project_id}", headers=auth(token_b))
    assert cross.status_code == 404

    own_list = await client.get("/v1/projects", headers=auth(token_b))
    assert all(p["id"] != project_id for p in own_list.json())
