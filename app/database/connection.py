"""
Database Connection Management

Handles PostgreSQL connections using SQLAlchemy with async support.
Implements connection pooling for high concurrency and proper
resource management for production environments.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """
    Database connection manager for high-concurrency applications.
    
    Provides both async and sync database engines with proper
    connection pooling and health monitoring.
    """
    
    def __init__(self):
        self._async_engine: Optional[AsyncEngine] = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize database connections and start health monitoring."""
        await self._create_engines()
        await self._setup_session_factories()
        await self._start_health_monitoring()
        logger.info("Database manager initialized successfully")
    
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._async_engine:
            await self._async_engine.dispose()
        
        if self._sync_engine:
            self._sync_engine.dispose()
        
        logger.info("Database connections closed")
    
    async def _create_engines(self) -> None:
        """Create async and sync database engines with optimized settings."""
        
        # Async engine for main application
        self._async_engine = create_async_engine(
            settings.database.url,
            poolclass=QueuePool,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle,
            pool_pre_ping=True,  # Validate connections before use
            echo=settings.debug,  # SQL logging in debug mode
            future=True,
        )
        
        # Sync engine for migrations and admin tasks
        self._sync_engine = create_engine(
            settings.database.sync_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=settings.debug,
            future=True,
        )
        
        # Add connection event listeners for monitoring
        self._setup_connection_events()
    
    def _setup_connection_events(self) -> None:
        """Setup database connection event listeners for monitoring."""
        
        @event.listens_for(self._sync_engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            """Log successful database connections."""
            logger.debug("Database connection established")
        
        @event.listens_for(self._sync_engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log connection checkouts from pool."""
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self._sync_engine, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            """Log connection checkins to pool."""
            logger.debug("Connection checked in to pool")
    
    async def _setup_session_factories(self) -> None:
        """Setup session factories for creating database sessions."""
        
        # Async session factory
        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        
        # Sync session factory
        self._sync_session_factory = sessionmaker(
            self._sync_engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    
    async def _start_health_monitoring(self) -> None:
        """Start background task for database health monitoring."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self) -> None:
        """Periodic health check for database connections."""
        while True:
            try:
                await asyncio.sleep(settings.monitoring.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Database health check failed", error=str(e))
    
    async def health_check(self) -> bool:
        """
        Perform database health check.
        
        Returns:
            bool: True if database is healthy, False otherwise.
        """
        try:
            async with self.get_async_session() as session:
                await session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with automatic cleanup.
        
        Usage:
            async with db_manager.get_async_session() as session:
                result = await session.execute(query)
        """
        if not self._async_session_factory:
            raise RuntimeError("Database manager not initialized")
        
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_sync_session(self) -> Session:
        """
        Get sync database session with automatic cleanup.
        
        Usage:
            with db_manager.get_sync_session() as session:
                result = session.execute(query)
        """
        if not self._sync_session_factory:
            raise RuntimeError("Database manager not initialized")
        
        with self._sync_session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
    
    @property
    def async_engine(self) -> AsyncEngine:
        """Get async database engine."""
        if not self._async_engine:
            raise RuntimeError("Database manager not initialized")
        return self._async_engine
    
    @property
    def sync_engine(self):
        """Get sync database engine."""
        if not self._sync_engine:
            raise RuntimeError("Database manager not initialized")
        return self._sync_engine
    
    async def get_pool_status(self) -> dict:
        """
        Get connection pool status for monitoring.
        
        Returns:
            dict: Pool status information including size, checked out connections, etc.
        """
        if not self._async_engine:
            return {}
        
        pool = self._async_engine.pool
        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
        }


# Global database manager instance
db_manager = DatabaseManager()


# Dependency for FastAPI routes
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting database session.
    
    Usage in route:
        async def create_user(
            user_data: UserCreate,
            db: AsyncSession = Depends(get_db_session)
        ):
            # Use db session here
    """
    async with db_manager.get_async_session() as session:
        yield session


# Convenience functions for direct usage
async def get_async_session() -> AsyncSession:
    """Get async session for direct usage (remember to close manually)."""
    if not db_manager._async_session_factory:
        raise RuntimeError("Database manager not initialized")
    return db_manager._async_session_factory()


def get_sync_session() -> Session:
    """Get sync session for direct usage (remember to close manually)."""
    if not db_manager._sync_session_factory:
        raise RuntimeError("Database manager not initialized")
    return db_manager._sync_session_factory()