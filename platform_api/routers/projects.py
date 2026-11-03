from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.errors import Conflict, NotFound
from common.scopes import SCOPE_PROJECT_ADMIN
from common.security import Principal
from platform_api.db import get_db
from platform_api.deps import audit, require_scope
from platform_api.models import Project, User
from platform_api.schemas import ProjectCreate, ProjectOut, ProjectUpdate

router = APIRouter(prefix="/v1/projects", tags=["projects"])


async def _get_project(db: AsyncSession, principal: Principal, project_id: str) -> Project:
    # 关键点：查询条件里必须带 tenant_id，不能先按 id 查再判断归属
    project = await db.scalar(
        select(Project).where(Project.id == project_id, Project.tenant_id == principal.tenant_id)
    )
    if project is None:
        raise NotFound("项目不存在")
    return project


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    principal: Principal = Depends(require_scope("chat:read")),
    db: AsyncSession = Depends(get_db),
) -> list[ProjectOut]:
    rows = await db.scalars(
        select(Project).where(Project.tenant_id == principal.tenant_id).order_by(Project.created_at)
    )
    return [ProjectOut.model_validate(p) for p in rows]


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: ProjectCreate,
    principal: Principal = Depends(require_scope(SCOPE_PROJECT_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> ProjectOut:
    exists = await db.scalar(
        select(Project).where(Project.tenant_id == principal.tenant_id, Project.slug == payload.slug)
    )
    if exists:
        raise Conflict(f"项目 slug 已存在：{payload.slug}")

    project = Project(
        tenant_id=principal.tenant_id,
        slug=payload.slug,
        name=payload.name,
        description=payload.description,
        graph_key=payload.graph_key,
        collection=payload.collection or f"{principal.tenant_id}_{payload.slug}",
        config=payload.config,
    )
    db.add(project)
    await db.flush()
    await audit(db, action="project.create", principal=principal, resource_type="project",
                resource_id=project.id, detail={"slug": project.slug})
    await db.commit()
    return ProjectOut.model_validate(project)


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: str,
    principal: Principal = Depends(require_scope("chat:read")),
    db: AsyncSession = Depends(get_db),
) -> ProjectOut:
    return ProjectOut.model_validate(await _get_project(db, principal, project_id))


@router.patch("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: str,
    payload: ProjectUpdate,
    principal: Principal = Depends(require_scope(SCOPE_PROJECT_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> ProjectOut:
    project = await _get_project(db, principal, project_id)
    changed = payload.model_dump(exclude_unset=True)
    for key, value in changed.items():
        setattr(project, key, value)
    await audit(db, action="project.update", principal=principal, resource_type="project",
                resource_id=project.id, detail={"fields": list(changed)})
    await db.commit()
    return ProjectOut.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    principal: Principal = Depends(require_scope(SCOPE_PROJECT_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> Response:
    project = await _get_project(db, principal, project_id)
    project.is_active = False  # 软删除：审计链不能断
    await audit(db, action="project.delete", principal=principal, resource_type="project",
                resource_id=project.id)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/by-slug/{slug}", response_model=ProjectOut)
async def get_project_by_slug(
    slug: str,
    principal: Principal = Depends(require_scope("chat:read")),
    db: AsyncSession = Depends(get_db),
) -> ProjectOut:
    project = await db.scalar(
        select(Project).where(Project.tenant_id == principal.tenant_id, Project.slug == slug)
    )
    if project is None:
        raise NotFound("项目不存在")
    return ProjectOut.model_validate(project)


@router.get("/{project_id}/members", response_model=list[dict])
async def list_members(
    project_id: str,
    principal: Principal = Depends(require_scope(SCOPE_PROJECT_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    await _get_project(db, principal, project_id)
    users = await db.scalars(select(User).where(User.tenant_id == principal.tenant_id))
    return [
        {"id": u.id, "email": u.email, "roles": u.roles, "scopes": u.scopes, "is_active": u.is_active}
        for u in users
    ]
