"""
Pytest Configuration and Fixtures

Provides common fixtures and configuration for all tests
in the application, including database setup, Redis mocking,
and test client creation.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

# Set testing environment before importing app modules
os.environ["TESTING"] = "true"
os.environ["ENVIRONMENT"] = "testing"

from app.config import settings
from app.database.base import DeclarativeBase
from app.database.connection import db_manager
from app.core.redis import redis_manager
from app.main import app


# Test database URLs (using SQLite for faster tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
TEST_SYNC_DATABASE_URL = "sqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def async_engine():
    """Create async database engine for testing."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(DeclarativeBase.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()
    
    # Remove test database file
    try:
        os.remove("./test.db")
    except FileNotFoundError:
        pass


@pytest_asyncio.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for testing."""
    async_session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_redis():
    """Mock Redis manager for testing."""
    redis_mock = Mock()
    redis_mock.set_cache = AsyncMock(return_value=True)
    redis_mock.get_cache = AsyncMock(return_value=None)
    redis_mock.delete_cache = AsyncMock(return_value=True)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.rate_limit_check = AsyncMock(return_value={
        "allowed": True,
        "remaining": 100,
        "reset_time": 60,
        "current_count": 0,
        "limit": 100,
    })
    redis_mock.health_check = AsyncMock(return_value=True)
    
    return redis_mock


@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram bot for testing."""
    bot_mock = Mock()
    bot_mock.send_message = AsyncMock()
    bot_mock.edit_message_text = AsyncMock()
    bot_mock.delete_message = AsyncMock()
    bot_mock.get_me = AsyncMock(return_value=Mock(
        id=12345,
        is_bot=True,
        first_name="Test Bot",
        username="test_bot"
    ))
    
    return bot_mock


@pytest.fixture
def test_client() -> TestClient:
    """Create FastAPI test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client():
    """Create async HTTP client for testing."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "telegram_id": 123456789,
        "username": "testuser",
        "first_name": "Test",
        "last_name": "User",
        "language_code": "en",
        "is_bot": False,
        "is_premium": False,
    }


@pytest.fixture
def sample_message_data():
    """Sample Telegram message data for testing."""
    return {
        "message_id": 1,
        "from": {
            "id": 123456789,
            "is_bot": False,
            "first_name": "Test",
            "last_name": "User",
            "username": "testuser",
            "language_code": "en"
        },
        "chat": {
            "id": 123456789,
            "first_name": "Test",
            "last_name": "User",
            "username": "testuser",
            "type": "private"
        },
        "date": 1640995200,
        "text": "/start"
    }


@pytest.fixture
def sample_update_data(sample_message_data):
    """Sample Telegram update data for testing."""
    return {
        "update_id": 1,
        "message": sample_message_data
    }


class MockMLModel:
    """Mock ML model for testing."""
    
    def __init__(self):
        self.device = "cpu"
        self.model_name = "test-model"
    
    async def predict(self, text: str):
        """Mock prediction."""
        return {
            "sentiment": "positive",
            "confidence": 0.95,
            "embeddings": [0.1, 0.2, 0.3]
        }
    
    async def analyze_personality(self, texts: list):
        """Mock personality analysis."""
        return {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "neuroticism": 0.3
        }


@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing."""
    return MockMLModel()


@pytest.fixture(autouse=True)
def setup_test_environment(mock_redis, mock_ml_model):
    """Setup test environment with mocked dependencies."""
    # Mock external dependencies
    import app.core.redis
    import app.services.ml
    
    # Replace Redis manager
    app.core.redis.redis_manager = mock_redis
    
    # Mock ML services
    if hasattr(app.services.ml, 'ml_service'):
        app.services.ml.ml_service.model = mock_ml_model


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "ml: mark test as requiring ML models")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to all tests by default
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add 'slow' marker to tests with 'slow' in the name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add 'integration' marker to tests in integration folders
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add 'ml' marker to ML-related tests
        if "ml" in item.name.lower() or "ml" in str(item.fspath):
            item.add_marker(pytest.mark.ml)


@pytest.fixture
def anyio_backend():
    """Use asyncio backend for anyio tests."""
    return "asyncio"