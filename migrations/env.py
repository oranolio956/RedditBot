"""
Alembic Migration Environment

This module provides the migration environment for database schema changes.
It supports both online (with database connection) and offline (SQL generation) modes.
"""

import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Import your models and configuration
from app.config import settings
from app.database.base import DeclarativeBase

# Alembic Config object for access to .ini file values
config = context.config

# Interpret the config file for logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for auto-generation
target_metadata = DeclarativeBase.metadata

# Override sqlalchemy.url from settings
config.set_main_option("sqlalchemy.url", settings.database.sync_url)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,  # For SQLite compatibility
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with an active database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,  # For SQLite compatibility
        # Include object naming conventions for consistent naming
        include_object=include_object,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def include_object(object, name, type_, reflected, compare_to):
    """
    Should we include this object in the migration?
    
    This function allows filtering of database objects during migration
    generation. You can exclude certain tables, columns, or indexes.
    """
    # Skip temporary tables
    if type_ == "table" and name.startswith("temp_"):
        return False
    
    # Skip certain system tables (if any)
    if type_ == "table" and name in ["spatial_ref_sys"]:
        return False
    
    return True


def include_name(name, type_, parent_names):
    """
    Should we include this name in the migration?
    
    This function allows filtering based on naming patterns.
    """
    # Skip temporary objects
    if name and name.startswith("temp_"):
        return False
    
    return True


async def run_async_migrations() -> None:
    """Run migrations asynchronously."""
    connectable = create_async_engine(
        settings.database.url,
        poolclass=pool.NullPool,  # No connection pooling for migrations
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context. For async databases, we use asyncio to handle the
    asynchronous database operations.
    """
    asyncio.run(run_async_migrations())


# Determine which mode to run migrations in
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()