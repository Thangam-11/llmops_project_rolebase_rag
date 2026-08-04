"""
Admin router — cache management and other operational endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from models.model import User, UserRole
from auth.dependencies import get_current_user
from api_services.routers.chat import get_rag_chain
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Raises 403 unless the current user's role is admin."""
    if user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


@router.post("/cache/invalidate")
async def invalidate_cache(
    user: User = Depends(require_admin),
):
    chain = get_rag_chain()

    if not chain._redis_cache.available:          # ← was chain._cache
        return {
            "invalidated_keys": 0,
            "message": "Cache is not available/connected — nothing to invalidate",
        }

    count = chain._redis_cache.invalidate_all()    # ← was chain._cache
    logger.info(f"Cache invalidated by {user.email} | {count} keys removed")

    return {
        "invalidated_keys": count,
        "message": f"Cleared {count} cached responses",
    }


@router.get("/cache/status")
async def cache_status(
    user: User = Depends(require_admin),
):
    chain = get_rag_chain()

    return {
        "cache_enabled": chain._redis_cache._enabled,     # ← was chain._cache
        "cache_available": chain._redis_cache.available,   # ← was chain._cache
        "ttl_seconds": chain._redis_cache._ttl,             # ← was chain._cache
    }