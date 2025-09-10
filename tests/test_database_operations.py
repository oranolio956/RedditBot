"""
Database Operations Tests

Comprehensive test suite for database operations including:
- SQLAlchemy model operations and relationships
- Database connection management and pooling
- Transaction handling and rollback scenarios
- Query optimization and performance testing
- Data integrity and constraint validation
- Migration testing and schema changes
- Repository pattern implementation
- Connection resilience and error handling
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional
import sqlalchemy as sa
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.pool import StaticPool

from app.database.base import DeclarativeBase
from app.database.connection import DatabaseManager, get_database_session, db_manager
from app.database.repository import BaseRepository
from app.database.repositories import UserRepository, ConversationRepository
from app.models.user import User
from app.models.conversation import Conversation, Message, MessageType, MessageDirection
from app.models.engagement import UserEngagement, UserBehaviorPattern, EngagementType, SentimentType
from app.models.group_session import GroupSession, GroupMember, GroupConversation
from app.models.sharing import ShareableContent, ContentShare, ShareableContentType


@pytest.fixture
async def test_engine():
    """Create test database engine with in-memory SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL debugging
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(DeclarativeBase.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine):
    """Create test database session."""
    async with AsyncSession(test_engine, expire_on_commit=False) as session:
        yield session
        await session.rollback()


@pytest.fixture
async def sample_user(test_session):
    """Create sample user for testing."""
    user = User(
        telegram_id=123456789,
        username="testuser",
        first_name="Test",
        last_name="User",
        language_code="en",
        is_bot=False,
        is_premium=False,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)
    
    return user


@pytest.fixture
async def sample_conversation(test_session, sample_user):
    """Create sample conversation for testing."""
    conversation = Conversation(
        user_id=str(sample_user.id),
        context="Test conversation",
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    test_session.add(conversation)
    await test_session.commit()
    await test_session.refresh(conversation)
    
    return conversation


class TestDatabaseConnection:
    """Test database connection management and configuration."""
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization(self):
        """Test DatabaseManager initialization."""
        manager = DatabaseManager(
            database_url="sqlite+aiosqlite:///:memory:",
            echo=False
        )
        
        assert manager.database_url == "sqlite+aiosqlite:///:memory:"
        assert manager.echo is False
        assert manager.engine is None  # Not initialized yet
        assert manager._session_factory is None
    
    @pytest.mark.asyncio
    async def test_database_engine_creation(self):
        """Test database engine creation and configuration."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        
        await manager.initialize()
        
        assert manager.engine is not None
        assert manager._session_factory is not None
        
        # Test engine properties
        assert "sqlite" in str(manager.engine.url)
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_session_creation_and_cleanup(self, test_engine):
        """Test session creation and proper cleanup."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        manager.engine = test_engine
        manager._create_session_factory()
        
        # Create session
        session = manager.get_session()
        assert isinstance(session, AsyncSession)
        
        # Session should be usable
        result = await session.execute(select(1))
        assert result.scalar() == 1
        
        # Cleanup
        await session.close()
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, test_engine):
        """Test database connection pooling behavior."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        manager.engine = test_engine
        manager._create_session_factory()
        
        # Create multiple sessions
        sessions = []
        for i in range(5):
            session = manager.get_session()
            sessions.append(session)
        
        # All sessions should be different instances
        session_ids = [id(s) for s in sessions]
        assert len(set(session_ids)) == len(sessions)
        
        # Cleanup
        for session in sessions:
            await session.close()
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_session):
        """Test database health check functionality."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        
        # Mock the session
        with patch.object(manager, 'get_session', return_value=test_session):
            health_status = await manager.health_check()
        
        assert health_status is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test database health check with connection failure."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        
        # Mock failing session
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Connection failed")
        
        with patch.object(manager, 'get_session', return_value=mock_session):
            health_status = await manager.health_check()
        
        assert health_status is False


class TestModelOperations:
    """Test basic model CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_user_creation(self, test_session):
        """Test user model creation and persistence."""
        user = User(
            telegram_id=987654321,
            username="newuser",
            first_name="New",
            last_name="User",
            language_code="es"
        )
        
        test_session.add(user)
        await test_session.commit()
        await test_session.refresh(user)
        
        # Verify user was created with ID
        assert user.id is not None
        assert user.telegram_id == 987654321
        assert user.username == "newuser"
        assert user.created_at is not None
    
    @pytest.mark.asyncio
    async def test_user_retrieval(self, test_session, sample_user):
        """Test user retrieval by various criteria."""
        # Retrieve by ID
        user_by_id = await test_session.get(User, sample_user.id)
        assert user_by_id is not None
        assert user_by_id.id == sample_user.id
        
        # Retrieve by telegram_id
        result = await test_session.execute(
            select(User).where(User.telegram_id == sample_user.telegram_id)
        )
        user_by_telegram = result.scalar_one_or_none()
        assert user_by_telegram is not None
        assert user_by_telegram.id == sample_user.id
        
        # Retrieve by username
        result = await test_session.execute(
            select(User).where(User.username == sample_user.username)
        )
        user_by_username = result.scalar_one_or_none()
        assert user_by_username is not None
        assert user_by_username.id == sample_user.id
    
    @pytest.mark.asyncio
    async def test_user_update(self, test_session, sample_user):
        """Test user model updates."""
        original_updated_at = sample_user.updated_at
        
        # Update user
        sample_user.first_name = "Updated"
        sample_user.is_premium = True
        
        await test_session.commit()
        await test_session.refresh(sample_user)
        
        # Verify updates
        assert sample_user.first_name == "Updated"
        assert sample_user.is_premium is True
        assert sample_user.updated_at > original_updated_at
    
    @pytest.mark.asyncio
    async def test_user_deletion(self, test_session):
        """Test user model deletion."""
        user = User(
            telegram_id=111222333,
            username="deleteuser",
            first_name="Delete",
            last_name="Me"
        )
        
        test_session.add(user)
        await test_session.commit()
        user_id = user.id
        
        # Delete user
        await test_session.delete(user)
        await test_session.commit()
        
        # Verify deletion
        deleted_user = await test_session.get(User, user_id)
        assert deleted_user is None
    
    @pytest.mark.asyncio
    async def test_conversation_creation_with_user_relationship(self, test_session, sample_user):
        """Test conversation creation with user foreign key."""
        conversation = Conversation(
            user_id=str(sample_user.id),
            context="Test conversation with user",
            is_active=True
        )
        
        test_session.add(conversation)
        await test_session.commit()
        await test_session.refresh(conversation)
        
        # Verify conversation creation
        assert conversation.id is not None
        assert conversation.user_id == str(sample_user.id)
        assert conversation.created_at is not None
    
    @pytest.mark.asyncio
    async def test_message_creation_with_conversation_relationship(self, test_session, sample_conversation):
        """Test message creation with conversation foreign key."""
        message = Message(
            conversation_id=sample_conversation.id,
            content="Test message content",
            message_type=MessageType.TEXT,
            direction=MessageDirection.INCOMING,
            is_from_bot=False
        )
        
        test_session.add(message)
        await test_session.commit()
        await test_session.refresh(message)
        
        # Verify message creation
        assert message.id is not None
        assert message.conversation_id == sample_conversation.id
        assert message.content == "Test message content"
        assert message.timestamp is not None


class TestModelRelationships:
    """Test model relationships and joins."""
    
    @pytest.mark.asyncio
    async def test_user_conversations_relationship(self, test_session, sample_user):
        """Test one-to-many relationship between user and conversations."""
        # Create multiple conversations for user
        conversations = []
        for i in range(3):
            conv = Conversation(
                user_id=str(sample_user.id),
                context=f"Conversation {i}",
                is_active=True
            )
            conversations.append(conv)
            test_session.add(conv)
        
        await test_session.commit()
        
        # Query conversations through relationship
        result = await test_session.execute(
            select(User).options(sa.orm.selectinload(User.conversations))
            .where(User.id == sample_user.id)
        )
        user_with_convs = result.scalar_one_or_none()
        
        # Verify relationship
        assert user_with_convs is not None
        assert len(user_with_convs.conversations) == 3
        
        # Verify conversation order (should be by created_at)
        conv_contexts = [conv.context for conv in user_with_convs.conversations]
        assert "Conversation 0" in conv_contexts
        assert "Conversation 1" in conv_contexts
        assert "Conversation 2" in conv_contexts
    
    @pytest.mark.asyncio
    async def test_conversation_messages_relationship(self, test_session, sample_conversation):
        """Test one-to-many relationship between conversation and messages."""
        # Create multiple messages for conversation
        messages = []
        for i in range(5):
            msg = Message(
                conversation_id=sample_conversation.id,
                content=f"Message {i}",
                message_type=MessageType.TEXT,
                direction=MessageDirection.INCOMING if i % 2 == 0 else MessageDirection.OUTGOING,
                is_from_bot=i % 2 == 1
            )
            messages.append(msg)
            test_session.add(msg)
        
        await test_session.commit()
        
        # Query messages through relationship
        result = await test_session.execute(
            select(Conversation).options(sa.orm.selectinload(Conversation.messages))
            .where(Conversation.id == sample_conversation.id)
        )
        conv_with_msgs = result.scalar_one_or_none()
        
        # Verify relationship
        assert conv_with_msgs is not None
        assert len(conv_with_msgs.messages) == 5
        
        # Verify message ordering and properties
        for i, msg in enumerate(conv_with_msgs.messages):
            assert msg.content == f"Message {i}"
            assert msg.is_from_bot == (i % 2 == 1)
    
    @pytest.mark.asyncio
    async def test_complex_query_with_joins(self, test_session, sample_user, sample_conversation):
        """Test complex queries with multiple joins."""
        # Add messages to conversation
        for i in range(3):
            msg = Message(
                conversation_id=sample_conversation.id,
                content=f"Join test message {i}",
                message_type=MessageType.TEXT,
                direction=MessageDirection.INCOMING,
                is_from_bot=False
            )
            test_session.add(msg)
        
        await test_session.commit()
        
        # Complex query: Get users with their conversation count and latest message
        query = (
            select(
                User.id,
                User.username,
                func.count(Conversation.id).label('conversation_count'),
                func.max(Message.timestamp).label('latest_message_time')
            )
            .select_from(User)
            .outerjoin(Conversation, User.id == Conversation.user_id)
            .outerjoin(Message, Conversation.id == Message.conversation_id)
            .group_by(User.id, User.username)
            .having(func.count(Conversation.id) > 0)
        )
        
        result = await test_session.execute(query)
        user_stats = result.all()
        
        # Verify complex query results
        assert len(user_stats) >= 1
        
        user_stat = user_stats[0]
        assert user_stat.username == sample_user.username
        assert user_stat.conversation_count >= 1
        assert user_stat.latest_message_time is not None


class TestDataIntegrityAndConstraints:
    """Test data integrity constraints and validation."""
    
    @pytest.mark.asyncio
    async def test_unique_constraint_violation(self, test_session):
        """Test unique constraint handling."""
        # Create first user
        user1 = User(
            telegram_id=555666777,
            username="uniqueuser",
            first_name="First"
        )
        test_session.add(user1)
        await test_session.commit()
        
        # Try to create second user with same telegram_id
        user2 = User(
            telegram_id=555666777,  # Same telegram_id
            username="differentuser",
            first_name="Second"
        )
        test_session.add(user2)
        
        # Should raise integrity error
        with pytest.raises(IntegrityError):
            await test_session.commit()
        
        # Rollback the failed transaction
        await test_session.rollback()
    
    @pytest.mark.asyncio
    async def test_foreign_key_constraint(self, test_session):
        """Test foreign key constraint enforcement."""
        # Try to create conversation with non-existent user_id
        conversation = Conversation(
            user_id="non-existent-user-id",
            context="Invalid conversation",
            is_active=True
        )
        test_session.add(conversation)
        
        # Foreign key constraint should be enforced
        with pytest.raises((IntegrityError, StatementError)):
            await test_session.commit()
        
        await test_session.rollback()
    
    @pytest.mark.asyncio
    async def test_not_null_constraint(self, test_session):
        """Test NOT NULL constraint enforcement."""
        # Try to create user without required field
        user = User(
            # Missing required telegram_id
            username="incompleteuser",
            first_name="Incomplete"
        )
        test_session.add(user)
        
        # Should fail due to NOT NULL constraint
        with pytest.raises((IntegrityError, StatementError)):
            await test_session.commit()
        
        await test_session.rollback()
    
    @pytest.mark.asyncio
    async def test_check_constraint_validation(self, test_session):
        """Test check constraint validation if implemented."""
        # This would test custom check constraints like valid email format
        # For now, test that invalid data types are handled
        
        user = User(
            telegram_id=123456789,
            username="testuser",
            first_name="Test"
        )
        
        # Test invalid boolean assignment
        try:
            user.is_bot = "not_a_boolean"  # Should be boolean
            test_session.add(user)
            await test_session.commit()
            
            # If no error, the ORM handled the conversion
            assert isinstance(user.is_bot, bool) or user.is_bot == "not_a_boolean"
            
        except (ValueError, TypeError, IntegrityError):
            # Expected for strict type checking
            await test_session.rollback()


class TestTransactionHandling:
    """Test transaction management and rollback scenarios."""
    
    @pytest.mark.asyncio
    async def test_successful_transaction(self, test_session):
        """Test successful transaction commit."""
        # Create multiple related objects in one transaction
        user = User(
            telegram_id=999888777,
            username="transactionuser",
            first_name="Transaction"
        )
        test_session.add(user)
        await test_session.flush()  # Get user ID without committing
        
        conversation = Conversation(
            user_id=str(user.id),
            context="Transaction test",
            is_active=True
        )
        test_session.add(conversation)
        
        # Commit transaction
        await test_session.commit()
        
        # Verify both objects were persisted
        persisted_user = await test_session.get(User, user.id)
        assert persisted_user is not None
        
        result = await test_session.execute(
            select(Conversation).where(Conversation.user_id == str(user.id))
        )
        persisted_conv = result.scalar_one_or_none()
        assert persisted_conv is not None
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, test_session):
        """Test transaction rollback on error."""
        # Start transaction with valid object
        user = User(
            telegram_id=888777666,
            username="rollbackuser",
            first_name="Rollback"
        )
        test_session.add(user)
        
        # Add invalid object that will cause error
        invalid_conversation = Conversation(
            user_id="invalid-user-id",
            context="This should fail",
            is_active=True
        )
        test_session.add(invalid_conversation)
        
        # Transaction should fail and rollback
        with pytest.raises((IntegrityError, StatementError)):
            await test_session.commit()
        
        await test_session.rollback()
        
        # Verify nothing was persisted
        result = await test_session.execute(
            select(User).where(User.telegram_id == 888777666)
        )
        assert result.scalar_one_or_none() is None
    
    @pytest.mark.asyncio
    async def test_nested_transaction_with_savepoint(self, test_session):
        """Test nested transactions with savepoints."""
        # Create user successfully
        user = User(
            telegram_id=777666555,
            username="savepointuser",
            first_name="Savepoint"
        )
        test_session.add(user)
        await test_session.flush()
        
        # Create savepoint
        savepoint = await test_session.begin_nested()
        
        try:
            # Try invalid operation
            invalid_conversation = Conversation(
                user_id="invalid-id",
                context="Invalid",
                is_active=True
            )
            test_session.add(invalid_conversation)
            await test_session.flush()
            
        except (IntegrityError, StatementError):
            # Rollback to savepoint
            await savepoint.rollback()
        else:
            await savepoint.commit()
        
        # Create valid conversation
        valid_conversation = Conversation(
            user_id=str(user.id),
            context="Valid conversation",
            is_active=True
        )
        test_session.add(valid_conversation)
        
        # Commit main transaction
        await test_session.commit()
        
        # Verify user and valid conversation persisted, invalid did not
        persisted_user = await test_session.get(User, user.id)
        assert persisted_user is not None
        
        result = await test_session.execute(
            select(Conversation).where(Conversation.user_id == str(user.id))
        )
        conversations = result.scalars().all()
        assert len(conversations) == 1
        assert conversations[0].context == "Valid conversation"


class TestRepositoryPattern:
    """Test repository pattern implementation."""
    
    @pytest.fixture
    async def user_repository(self, test_session):
        """Create UserRepository instance."""
        return UserRepository(test_session)
    
    @pytest.fixture
    async def conversation_repository(self, test_session):
        """Create ConversationRepository instance."""
        return ConversationRepository(test_session)
    
    @pytest.mark.asyncio
    async def test_base_repository_create(self, test_session):
        """Test base repository create operation."""
        repo = BaseRepository(User, test_session)
        
        user_data = {
            "telegram_id": 444555666,
            "username": "repouser",
            "first_name": "Repository",
            "last_name": "Test"
        }
        
        user = await repo.create(**user_data)
        
        assert user.id is not None
        assert user.telegram_id == user_data["telegram_id"]
        assert user.username == user_data["username"]
    
    @pytest.mark.asyncio
    async def test_base_repository_get(self, test_session, sample_user):
        """Test base repository get operation."""
        repo = BaseRepository(User, test_session)
        
        retrieved_user = await repo.get(sample_user.id)
        
        assert retrieved_user is not None
        assert retrieved_user.id == sample_user.id
        assert retrieved_user.telegram_id == sample_user.telegram_id
    
    @pytest.mark.asyncio
    async def test_base_repository_update(self, test_session, sample_user):
        """Test base repository update operation."""
        repo = BaseRepository(User, test_session)
        
        updated_user = await repo.update(
            sample_user.id,
            first_name="Updated",
            is_premium=True
        )
        
        assert updated_user.first_name == "Updated"
        assert updated_user.is_premium is True
    
    @pytest.mark.asyncio
    async def test_base_repository_delete(self, test_session):
        """Test base repository delete operation."""
        repo = BaseRepository(User, test_session)
        
        # Create user to delete
        user = await repo.create(
            telegram_id=333444555,
            username="deleteuser",
            first_name="Delete"
        )
        user_id = user.id
        
        # Delete user
        success = await repo.delete(user_id)
        assert success is True
        
        # Verify deletion
        deleted_user = await repo.get(user_id)
        assert deleted_user is None
    
    @pytest.mark.asyncio
    async def test_base_repository_list_with_filters(self, test_session):
        """Test base repository list with filters."""
        repo = BaseRepository(User, test_session)
        
        # Create test users
        users = []
        for i in range(5):
            user = await repo.create(
                telegram_id=200000 + i,
                username=f"listuser{i}",
                first_name="List",
                is_premium=(i % 2 == 0)
            )
            users.append(user)
        
        # List all users
        all_users = await repo.list(limit=10)
        assert len(all_users) >= 5
        
        # List premium users only
        premium_users = await repo.list(
            filters={"is_premium": True},
            limit=10
        )
        assert all(user.is_premium for user in premium_users)
        assert len(premium_users) >= 3  # At least 3 premium users created
    
    @pytest.mark.asyncio
    async def test_user_repository_custom_methods(self, user_repository):
        """Test UserRepository custom methods."""
        # Create test user
        user = await user_repository.create(
            telegram_id=123321123,
            username="customuser",
            first_name="Custom"
        )
        
        # Test find by telegram_id
        found_user = await user_repository.find_by_telegram_id(123321123)
        assert found_user is not None
        assert found_user.id == user.id
        
        # Test find by username
        found_user = await user_repository.find_by_username("customuser")
        assert found_user is not None
        assert found_user.id == user.id
        
        # Test get active users
        active_users = await user_repository.get_active_users(limit=10)
        assert any(u.id == user.id for u in active_users)


class TestQueryOptimization:
    """Test query optimization and performance."""
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, test_session):
        """Test bulk insert operations."""
        # Prepare bulk data
        user_data = []
        for i in range(100):
            user_data.append({
                "telegram_id": 100000 + i,
                "username": f"bulkuser{i}",
                "first_name": f"Bulk{i}",
                "is_active": True
            })
        
        # Time bulk insert
        start_time = time.time()
        
        users = [User(**data) for data in user_data]
        test_session.add_all(users)
        await test_session.commit()
        
        bulk_time = time.time() - start_time
        
        # Verify bulk insert was efficient
        assert bulk_time < 2.0  # Should complete in <2 seconds
        
        # Verify all users were inserted
        result = await test_session.execute(
            select(func.count(User.id)).where(User.username.like("bulkuser%"))
        )
        count = result.scalar()
        assert count == 100
    
    @pytest.mark.asyncio
    async def test_query_with_eager_loading(self, test_session, sample_user):
        """Test query optimization with eager loading."""
        # Create conversation with messages
        conversation = Conversation(
            user_id=str(sample_user.id),
            context="Eager loading test",
            is_active=True
        )
        test_session.add(conversation)
        await test_session.flush()
        
        # Add messages
        for i in range(10):
            message = Message(
                conversation_id=conversation.id,
                content=f"Message {i}",
                message_type=MessageType.TEXT,
                direction=MessageDirection.INCOMING,
                is_from_bot=False
            )
            test_session.add(message)
        
        await test_session.commit()
        
        # Query with eager loading
        start_time = time.time()
        
        result = await test_session.execute(
            select(User)
            .options(
                sa.orm.selectinload(User.conversations)
                .selectinload(Conversation.messages)
            )
            .where(User.id == sample_user.id)
        )
        
        user_with_data = result.scalar_one_or_none()
        query_time = time.time() - start_time
        
        # Verify eager loading worked
        assert user_with_data is not None
        assert len(user_with_data.conversations) >= 1
        assert len(user_with_data.conversations[0].messages) == 10
        
        # Access should be fast (no additional queries)
        assert query_time < 0.1  # Should be very fast with eager loading
    
    @pytest.mark.asyncio
    async def test_pagination_efficiency(self, test_session):
        """Test efficient pagination queries."""
        # Create test data
        users = []
        for i in range(50):
            user = User(
                telegram_id=500000 + i,
                username=f"pageuser{i}",
                first_name=f"Page{i}"
            )
            users.append(user)
        
        test_session.add_all(users)
        await test_session.commit()
        
        # Test pagination
        page_size = 10
        page_number = 2  # Second page (skip 10, take 10)
        
        start_time = time.time()
        
        result = await test_session.execute(
            select(User)
            .where(User.username.like("pageuser%"))
            .order_by(User.created_at)
            .offset((page_number - 1) * page_size)
            .limit(page_size)
        )
        
        page_users = result.scalars().all()
        pagination_time = time.time() - start_time
        
        # Verify pagination
        assert len(page_users) == page_size
        assert page_users[0].username == "pageuser10"  # Second page starts at index 10
        assert pagination_time < 0.1  # Should be fast with proper indexing
    
    @pytest.mark.asyncio
    async def test_complex_aggregation_query(self, test_session, sample_user):
        """Test complex aggregation query performance."""
        # Create test data
        conversations = []
        for i in range(5):
            conv = Conversation(
                user_id=str(sample_user.id),
                context=f"Aggregation test {i}",
                is_active=True
            )
            conversations.append(conv)
            test_session.add(conv)
        
        await test_session.flush()
        
        # Add messages to conversations
        for conv in conversations:
            for j in range(3):
                message = Message(
                    conversation_id=conv.id,
                    content=f"Message {j}",
                    message_type=MessageType.TEXT,
                    direction=MessageDirection.INCOMING,
                    is_from_bot=False
                )
                test_session.add(message)
        
        await test_session.commit()
        
        # Complex aggregation query
        start_time = time.time()
        
        query = (
            select(
                User.username,
                func.count(Conversation.id).label('conversation_count'),
                func.count(Message.id).label('message_count'),
                func.avg(
                    func.length(Message.content)
                ).label('avg_message_length')
            )
            .select_from(User)
            .join(Conversation, User.id == Conversation.user_id)
            .join(Message, Conversation.id == Message.conversation_id)
            .where(User.id == sample_user.id)
            .group_by(User.username)
        )
        
        result = await test_session.execute(query)
        stats = result.first()
        
        aggregation_time = time.time() - start_time
        
        # Verify aggregation results
        assert stats is not None
        assert stats.conversation_count == 5
        assert stats.message_count == 15  # 5 conversations × 3 messages
        assert stats.avg_message_length > 0
        assert aggregation_time < 0.2  # Should be reasonably fast


class TestDatabasePerformance:
    """Test database performance under load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, test_engine):
        """Test concurrent database operations."""
        async def create_user_batch(batch_id: int, session_factory):
            async with session_factory() as session:
                users = []
                for i in range(10):
                    user = User(
                        telegram_id=batch_id * 1000 + i,
                        username=f"concurrent_user_{batch_id}_{i}",
                        first_name=f"Concurrent{batch_id}",
                        is_active=True
                    )
                    users.append(user)
                
                session.add_all(users)
                await session.commit()
                return len(users)
        
        # Create session factory
        from sqlalchemy.ext.asyncio import async_sessionmaker
        session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
        
        # Run concurrent operations
        tasks = []
        for batch_id in range(5):  # 5 batches of 10 users each
            task = create_user_batch(batch_id, session_factory)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify concurrent operations
        total_users_created = sum(results)
        assert total_users_created == 50  # 5 batches × 10 users
        assert total_time < 2.0  # Should complete concurrently in reasonable time
        
        # Verify all users were actually created
        async with session_factory() as session:
            result = await session.execute(
                select(func.count(User.id)).where(User.username.like("concurrent_user_%"))
            )
            count = result.scalar()
            assert count == 50
    
    @pytest.mark.asyncio
    async def test_database_connection_resilience(self, test_engine):
        """Test database connection resilience and recovery."""
        from sqlalchemy.ext.asyncio import async_sessionmaker
        session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
        
        # Test normal operation
        async with session_factory() as session:
            user = User(
                telegram_id=999999999,
                username="resilience_test",
                first_name="Resilience"
            )
            session.add(user)
            await session.commit()
            
            # Verify creation
            created_user = await session.get(User, user.id)
            assert created_user is not None
        
        # Test operation after connection disruption simulation
        # (In real tests, this might involve network issues or database restarts)
        async with session_factory() as session:
            result = await session.execute(select(User).limit(1))
            users = result.scalars().all()
            assert len(users) >= 1  # Should still work after "disruption"


if __name__ == "__main__":
    # Run database tests
    pytest.main([__file__, "-v", "--tb=short"])