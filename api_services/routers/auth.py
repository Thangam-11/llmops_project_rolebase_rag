"""
Auth router — with Prometheus metrics
"""
from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from models.database        import get_db
from models.model           import User
from auth.auth_service import AuthService
from auth.dependencies      import get_current_user
from src.monitoring.metrices import AUTH_LOGINS, AUTH_REGISTRATIONS, HTTP_REQUESTS
from api_services.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    RefreshRequest,
    LogoutRequest,
    TokenResponse,
    AccessTokenResponse,
    MeResponse,
)
from utils.logger_exceptions import get_logger

logger  = get_logger(__name__)
router  = APIRouter(prefix="/auth", tags=["Authentication"])
service = AuthService()


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    req: RegisterRequest,
    db:  AsyncSession = Depends(get_db),
):
    user = await service.register(
        email      = req.email,
        username   = req.username,
        password   = req.password,
        full_name  = req.full_name,
        department = req.department.value,
        role       = req.role.value,
        db         = db,
    )
    AUTH_REGISTRATIONS.inc()
    HTTP_REQUESTS.labels(method="POST", endpoint="/auth/register", status=201).inc()

    return {
        "message":    "Registered successfully",
        "username":   user.username,
        "email":      user.email,
        "department": user.department.value,
        "role":       user.role.value,
    }


@router.post("/login", response_model=TokenResponse)
async def login(
    req:     LoginRequest,
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    try:
        result = await service.login(
            email      = req.email,
            password   = req.password,
            db         = db,
            user_agent = request.headers.get("user-agent"),
            ip_address = request.client.host if request.client else None,
        )
        AUTH_LOGINS.labels(status="success").inc()
        HTTP_REQUESTS.labels(method="POST", endpoint="/auth/login", status=200).inc()
        return result

    except Exception:
        AUTH_LOGINS.labels(status="failed").inc()
        HTTP_REQUESTS.labels(method="POST", endpoint="/auth/login", status=401).inc()
        raise


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh(
    req: RefreshRequest,
    db:  AsyncSession = Depends(get_db),
):
    HTTP_REQUESTS.labels(method="POST", endpoint="/auth/refresh", status=200).inc()
    return await service.refresh(raw_token=req.refresh_token, db=db)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    req: LogoutRequest,
    db:  AsyncSession = Depends(get_db),
):
    HTTP_REQUESTS.labels(method="POST", endpoint="/auth/logout", status=204).inc()
    await service.logout(raw_token=req.refresh_token, db=db)


@router.get("/me", response_model=MeResponse)
async def me(user: User = Depends(get_current_user)):
    HTTP_REQUESTS.labels(method="GET", endpoint="/auth/me", status=200).inc()
    return {
        "id":                  str(user.id),
        "email":               user.email,
        "username":            user.username,
        "full_name":           user.full_name,
        "department":          user.department.value,
        "role":                user.role.value,
        "is_active":           user.is_active,
        "last_login":          str(user.last_login) if user.last_login else None,
        "allowed_collections": user.allowed_collections,
    }