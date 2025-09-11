"""
User Management API Endpoints

Provides REST API endpoints for user management including
CRUD operations, user statistics, and preferences.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate, UserResponse, UserStats
from app.core.auth import (
    get_current_user, require_permission, require_admin,
    AuthenticatedUser, Permission
)
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of users to return"),
    active_only: bool = Query(True, description="Return only active users"),
    current_user: AuthenticatedUser = Depends(require_permission(Permission.READ_USERS)),
    db: AsyncSession = Depends(get_db_session)
) -> List[UserResponse]:
    """
    Get list of users with pagination.
    
    Args:
        skip: Number of users to skip for pagination
        limit: Maximum number of users to return
        active_only: Filter for active users only
        db: Database session
        
    Returns:
        List of user objects
    """
    try:
        query = select(User)
        
        if active_only:
            query = query.where(User.is_active == True, User.is_deleted == False)
        
        query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
        
        result = await db.execute(query)
        users = result.scalars().all()
        
        return users
        
    except Exception as e:
        logger.error("Failed to get users", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve users")


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    current_user: AuthenticatedUser = Depends(require_permission(Permission.READ_USERS)),
    db: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    """
    Get user by ID.
    
    Args:
        user_id: User UUID
        db: Database session
        
    Returns:
        User object
    """
    try:
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve user")


@router.get("/telegram/{telegram_id}", response_model=UserResponse)
async def get_user_by_telegram_id(
    telegram_id: int,
    db: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    """
    Get user by Telegram ID.
    
    Args:
        telegram_id: Telegram user ID
        db: Database session
        
    Returns:
        User object
    """
    try:
        query = select(User).where(User.telegram_id == telegram_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user by telegram_id", telegram_id=telegram_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve user")


@router.post("/", response_model=UserResponse, status_code=201)
async def create_user(
    user_data: UserCreate,
    current_user: AuthenticatedUser = Depends(require_permission(Permission.WRITE_USERS)),
    db: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    """
    Create new user.
    
    Args:
        user_data: User creation data
        db: Database session
        
    Returns:
        Created user object
    """
    try:
        # Check if user already exists
        existing_query = select(User).where(User.telegram_id == user_data.telegram_id)
        existing_result = await db.execute(existing_query)
        existing_user = existing_result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=409,
                detail="User with this Telegram ID already exists"
            )
        
        # Create new user
        user = User(**user_data.model_dump())
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        logger.info("User created successfully", user_id=user.id, telegram_id=user.telegram_id)
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create user", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    current_user: AuthenticatedUser = Depends(require_permission(Permission.WRITE_USERS)),
    db: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    """
    Update user information.
    
    Args:
        user_id: User UUID
        user_data: User update data
        db: Database session
        
    Returns:
        Updated user object
    """
    try:
        # Get existing user
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if current user can update this user (self or admin)
        if not current_user.is_admin and current_user.user_id != user_id:
            raise HTTPException(
                status_code=403, 
                detail="You can only update your own profile"
            )
        
        # Update user with provided data
        update_data = user_data.model_dump(exclude_unset=True)
        user.update_from_dict(update_data)
        
        await db.commit()
        await db.refresh(user)
        
        logger.info("User updated successfully", user_id=user.id)
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user", user_id=user_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: UUID,
    hard_delete: bool = Query(False, description="Perform hard delete instead of soft delete"),
    current_user: AuthenticatedUser = Depends(require_permission(Permission.DELETE_USERS)),
    db: AsyncSession = Depends(get_db_session)
) -> None:
    """
    Delete user (soft delete by default).
    
    Args:
        user_id: User UUID
        hard_delete: Whether to perform hard delete
        db: Database session
    """
    try:
        # Get existing user
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if hard_delete:
            # Hard delete - remove from database
            await db.delete(user)
            logger.info("User hard deleted", user_id=user.id)
        else:
            # Soft delete - mark as deleted
            user.soft_delete()
            logger.info("User soft deleted", user_id=user.id)
        
        await db.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete user", user_id=user_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete user")


@router.post("/{user_id}/restore", response_model=UserResponse)
async def restore_user(
    user_id: UUID,
    current_user: AuthenticatedUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    """
    Restore soft-deleted user.
    
    Args:
        user_id: User UUID
        db: Database session
        
    Returns:
        Restored user object
    """
    try:
        # Get user including soft-deleted ones
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.is_deleted:
            raise HTTPException(status_code=409, detail="User is not deleted")
        
        # Restore user
        user.restore()
        await db.commit()
        await db.refresh(user)
        
        logger.info("User restored successfully", user_id=user.id)
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to restore user", user_id=user_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to restore user")


@router.get("/{user_id}/stats", response_model=UserStats)
async def get_user_stats(
    user_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> UserStats:
    """
    Get user statistics and interaction data.
    
    Args:
        user_id: User UUID
        db: Database session
        
    Returns:
        User statistics
    """
    try:
        # Get user
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get interaction stats
        stats = user.get_interaction_stats()
        
        return UserStats(
            user_id=user.id,
            telegram_id=user.telegram_id,
            **stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user stats", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve user stats")


@router.put("/{user_id}/preferences", response_model=UserResponse)
async def update_user_preferences(
    user_id: UUID,
    preferences: dict,
    db: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    """
    Update user preferences.
    
    Args:
        user_id: User UUID
        preferences: Preferences dictionary
        db: Database session
        
    Returns:
        Updated user object
    """
    try:
        # Get user
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update preferences
        for key, value in preferences.items():
            user.set_preference(key, value)
        
        await db.commit()
        await db.refresh(user)
        
        logger.info("User preferences updated", user_id=user.id)
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user preferences", user_id=user_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.get("/stats/overview")
async def get_users_overview(
    db: AsyncSession = Depends(get_db_session)
) -> dict:
    """
    Get overview statistics for all users.
    
    Args:
        db: Database session
        
    Returns:
        Overview statistics
    """
    try:
        # Total users
        total_query = select(func.count(User.id))
        total_result = await db.execute(total_query)
        total_users = total_result.scalar()
        
        # Active users
        active_query = select(func.count(User.id)).where(
            User.is_active == True,
            User.is_deleted == False
        )
        active_result = await db.execute(active_query)
        active_users = active_result.scalar()
        
        # New users (last 24 hours)
        from datetime import datetime, timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        new_query = select(func.count(User.id)).where(User.created_at >= yesterday)
        new_result = await db.execute(new_query)
        new_users = new_result.scalar()
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "new_users_24h": new_users,
            "inactive_users": total_users - active_users,
        }
        
    except Exception as e:
        logger.error("Failed to get users overview", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve overview")