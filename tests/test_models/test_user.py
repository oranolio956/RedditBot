"""
Test User Model

Tests for the User model including CRUD operations,
validation, and business logic methods.
"""

import pytest
from datetime import datetime, timedelta

from app.models.user import User


class TestUserModel:
    """Test cases for User model."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, db_session, sample_user_data):
        """Test creating a new user."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        assert user.id is not None
        assert user.telegram_id == sample_user_data["telegram_id"]
        assert user.username == sample_user_data["username"]
        assert user.first_name == sample_user_data["first_name"]
        assert user.created_at is not None
        assert user.updated_at is not None
    
    @pytest.mark.asyncio
    async def test_user_display_name(self, db_session):
        """Test user display name logic."""
        # User with full name
        user1 = User(
            telegram_id=123,
            first_name="John",
            last_name="Doe"
        )
        assert user1.get_display_name() == "John Doe"
        
        # User with first name only
        user2 = User(
            telegram_id=124,
            first_name="Jane"
        )
        assert user2.get_display_name() == "Jane"
        
        # User with username only
        user3 = User(
            telegram_id=125,
            username="johndoe"
        )
        assert user3.get_display_name() == "@johndoe"
        
        # User with only telegram_id
        user4 = User(telegram_id=126)
        assert user4.get_display_name() == "User 126"
    
    @pytest.mark.asyncio
    async def test_user_mention(self, db_session):
        """Test user mention generation."""
        # User with username
        user1 = User(
            telegram_id=123,
            username="johndoe",
            first_name="John"
        )
        assert user1.get_mention() == "@johndoe"
        
        # User without username
        user2 = User(
            telegram_id=124,
            first_name="Jane"
        )
        assert user2.get_mention() == "Jane"
        
        # User with only telegram_id
        user3 = User(telegram_id=125)
        assert user3.get_mention() == "User 125"
    
    @pytest.mark.asyncio
    async def test_update_activity(self, db_session, sample_user_data):
        """Test activity tracking."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        
        initial_message_count = user.message_count
        initial_command_count = user.command_count
        
        # Update with message activity
        user.update_activity("message")
        assert user.last_activity == "message"
        assert user.message_count == initial_message_count + 1
        assert user.command_count == initial_command_count
        
        # Update with command activity
        user.update_activity("command")
        assert user.last_activity == "command"
        assert user.command_count == initial_command_count + 1
    
    @pytest.mark.asyncio
    async def test_preferences(self, db_session, sample_user_data):
        """Test user preferences management."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        
        # Test setting preferences
        user.set_preference("language", "en")
        user.set_preference("notifications", True)
        
        # Test getting preferences
        assert user.get_preference("language") == "en"
        assert user.get_preference("notifications") is True
        assert user.get_preference("non_existent", "default") == "default"
        
        # Save and reload
        await db_session.commit()
        await db_session.refresh(user)
        
        # Verify preferences persisted
        assert user.get_preference("language") == "en"
        assert user.get_preference("notifications") is True
    
    @pytest.mark.asyncio
    async def test_new_user_check(self, db_session, sample_user_data):
        """Test new user detection."""
        # Create user
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Should be new user
        assert user.is_new_user() is True
        
        # Simulate old user by changing created_at
        user.created_at = datetime.utcnow() - timedelta(days=2)
        assert user.is_new_user() is False
    
    @pytest.mark.asyncio
    async def test_interaction_stats(self, db_session, sample_user_data):
        """Test interaction statistics."""
        user = User(**sample_user_data)
        user.message_count = 10
        user.command_count = 5
        user.last_activity = "message"
        
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        stats = user.get_interaction_stats()
        
        assert stats["total_messages"] == 10
        assert stats["total_commands"] == 5
        assert stats["last_activity"] == "message"
        assert stats["days_active"] >= 0
        assert "is_new_user" in stats
    
    @pytest.mark.asyncio
    async def test_soft_delete(self, db_session, sample_user_data):
        """Test soft delete functionality."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        
        assert user.is_deleted is False
        assert user.deleted_at is None
        
        # Soft delete
        user.soft_delete()
        assert user.is_deleted is True
        assert user.deleted_at is not None
        
        # Restore
        user.restore()
        assert user.is_deleted is False
        assert user.deleted_at is None
    
    @pytest.mark.asyncio
    async def test_to_dict(self, db_session, sample_user_data):
        """Test model to dictionary conversion."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        user_dict = user.to_dict()
        
        assert "id" in user_dict
        assert "telegram_id" in user_dict
        assert "username" in user_dict
        assert "created_at" in user_dict
        assert user_dict["telegram_id"] == sample_user_data["telegram_id"]
        
        # Test with exclusions
        user_dict_excluded = user.to_dict(exclude=["id", "created_at"])
        assert "id" not in user_dict_excluded
        assert "created_at" not in user_dict_excluded
        assert "telegram_id" in user_dict_excluded
    
    @pytest.mark.asyncio
    async def test_update_from_dict(self, db_session, sample_user_data):
        """Test updating model from dictionary."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        
        original_id = user.id
        original_created_at = user.created_at
        
        # Update from dict
        update_data = {
            "first_name": "Updated",
            "last_name": "Name",
            "is_premium": True,
            "id": "should_be_ignored",  # Should be excluded
            "created_at": datetime.utcnow(),  # Should be excluded
        }
        
        user.update_from_dict(update_data)
        
        assert user.first_name == "Updated"
        assert user.last_name == "Name"
        assert user.is_premium is True
        assert user.id == original_id  # Should not change
        assert user.created_at == original_created_at  # Should not change
    
    @pytest.mark.asyncio
    async def test_user_string_representation(self, sample_user_data):
        """Test string representation of user."""
        user = User(**sample_user_data)
        
        str_repr = str(user)
        assert "User(" in str_repr
        assert str(user.telegram_id) in str_repr
        assert user.first_name in str_repr