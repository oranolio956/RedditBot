"""
User Pydantic Schemas

Defines Pydantic models for user data validation,
serialization, and API request/response handling.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator


class UserBase(BaseModel):
    """Base user schema with common fields."""
    
    telegram_id: int = Field(..., description="Telegram user ID")
    username: Optional[str] = Field(None, max_length=32, description="Telegram username")
    first_name: Optional[str] = Field(None, max_length=64, description="User's first name")
    last_name: Optional[str] = Field(None, max_length=64, description="User's last name")
    language_code: Optional[str] = Field("en", max_length=10, description="Language code")
    is_bot: bool = Field(False, description="Whether user is a bot")
    is_premium: bool = Field(False, description="Whether user has Telegram Premium")
    
    @validator("telegram_id")
    def validate_telegram_id(cls, v):
        if v <= 0:
            raise ValueError("Telegram ID must be positive")
        return v
    
    @validator("username")
    def validate_username(cls, v):
        if v is not None:
            # Remove @ if present
            v = v.lstrip("@")
            # Validate format (alphanumeric and underscores only)
            if not v.replace("_", "").isalnum():
                raise ValueError("Username can only contain letters, numbers, and underscores")
        return v


class UserCreate(UserBase):
    """Schema for creating new users."""
    
    # All fields from UserBase are sufficient for creation
    pass


class UserUpdate(BaseModel):
    """Schema for updating existing users."""
    
    username: Optional[str] = Field(None, max_length=32)
    first_name: Optional[str] = Field(None, max_length=64)
    last_name: Optional[str] = Field(None, max_length=64)
    language_code: Optional[str] = Field(None, max_length=10)
    is_premium: Optional[bool] = None
    is_active: Optional[bool] = None
    is_blocked: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None
    
    @validator("username")
    def validate_username(cls, v):
        if v is not None:
            v = v.lstrip("@")
            if not v.replace("_", "").isalnum():
                raise ValueError("Username can only contain letters, numbers, and underscores")
        return v


class UserResponse(UserBase):
    """Schema for user responses."""
    
    id: UUID = Field(..., description="User UUID")
    is_active: bool = Field(..., description="Whether user is active")
    is_blocked: bool = Field(..., description="Whether user is blocked")
    is_deleted: bool = Field(..., description="Whether user is soft deleted")
    message_count: int = Field(..., description="Total message count")
    command_count: int = Field(..., description="Total command count")
    first_interaction: Optional[str] = Field(None, description="First interaction type")
    last_activity: Optional[str] = Field(None, description="Last activity type")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class UserStats(BaseModel):
    """Schema for user statistics."""
    
    user_id: UUID = Field(..., description="User UUID")
    telegram_id: int = Field(..., description="Telegram user ID")
    total_messages: int = Field(..., description="Total messages sent")
    total_commands: int = Field(..., description="Total commands used")
    last_activity: Optional[str] = Field(None, description="Last activity type")
    days_active: int = Field(..., description="Number of days since registration")
    is_new_user: bool = Field(..., description="Whether user registered recently")
    
    class Config:
        json_encoders = {
            UUID: lambda v: str(v),
        }


class UserPreferences(BaseModel):
    """Schema for user preferences."""
    
    language: Optional[str] = Field("en", description="Preferred language")
    notifications: Optional[bool] = Field(True, description="Enable notifications")
    timezone: Optional[str] = Field("UTC", description="User timezone")
    theme: Optional[str] = Field("default", description="UI theme preference")
    ml_features: Optional[bool] = Field(True, description="Enable ML features")
    
    class Config:
        extra = "allow"  # Allow additional preference fields


class UserList(BaseModel):
    """Schema for paginated user lists."""
    
    users: list[UserResponse] = Field(..., description="List of users")
    total: int = Field(..., description="Total number of users")
    skip: int = Field(..., description="Number of users skipped")
    limit: int = Field(..., description="Maximum users returned")
    has_more: bool = Field(..., description="Whether more users are available")
    
    @validator("has_more", always=True)
    def calculate_has_more(cls, v, values):
        if "total" in values and "skip" in values and "limit" in values:
            return values["skip"] + values["limit"] < values["total"]
        return False