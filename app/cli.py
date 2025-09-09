"""
Command Line Interface

Provides CLI commands for database migrations, bot management,
and maintenance tasks.
"""

import asyncio
import click
from typing import Optional

from app.config import settings
from app.database.connection import db_manager
from app.core.redis import redis_manager


@click.group()
def cli():
    """Telegram ML Bot CLI."""
    pass


@cli.command()
@click.option("--host", default=settings.host, help="Host to bind to")
@click.option("--port", default=settings.port, help="Port to bind to")
@click.option("--workers", default=settings.workers, help="Number of worker processes")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def start_server(host: str, port: int, workers: int, reload: bool):
    """Start the FastAPI server."""
    import uvicorn
    
    click.echo(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=1 if reload else workers,
        reload=reload,
        loop="uvloop" if not reload else "auto",
        log_level=settings.monitoring.log_level.lower(),
    )


@cli.command()
def start_bot():
    """Start the Telegram bot."""
    click.echo("Starting Telegram bot...")
    
    from app.bot.main import run_bot
    asyncio.run(run_bot())


@cli.command()
@click.option("--concurrency", default=4, help="Number of worker processes")
@click.option("--loglevel", default="info", help="Log level")
@click.option("--queues", default="default,ml_tasks", help="Comma-separated list of queues")
def start_worker(concurrency: int, loglevel: str, queues: str):
    """Start Celery worker."""
    import subprocess
    
    queue_list = queues.split(",")
    queue_args = ["-Q", ",".join(queue_list)]
    
    cmd = [
        "celery",
        "-A", "app.worker",
        "worker",
        "--loglevel", loglevel,
        "--concurrency", str(concurrency),
    ] + queue_args
    
    click.echo(f"Starting Celery worker with queues: {queue_list}")
    subprocess.run(cmd)


@cli.command()
@click.option("--loglevel", default="info", help="Log level")
def start_scheduler(loglevel: str):
    """Start Celery beat scheduler."""
    import subprocess
    
    cmd = [
        "celery",
        "-A", "app.worker",
        "beat",
        "--loglevel", loglevel,
        "--scheduler", "django_celery_beat.schedulers:DatabaseScheduler",
    ]
    
    click.echo("Starting Celery beat scheduler")
    subprocess.run(cmd)


@cli.group()
def db():
    """Database management commands."""
    pass


@db.command("init")
def init_db():
    """Initialize database with tables."""
    import asyncio
    from alembic.config import Config
    from alembic import command
    
    async def init():
        await db_manager.initialize()
        
        # Run Alembic migrations
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        
        await db_manager.close()
    
    click.echo("Initializing database...")
    asyncio.run(init())
    click.echo("Database initialized successfully!")


@db.command("migrate")
@click.option("--message", "-m", required=True, help="Migration message")
def create_migration(message: str):
    """Create new database migration."""
    from alembic.config import Config
    from alembic import command
    
    click.echo(f"Creating migration: {message}")
    
    alembic_cfg = Config("alembic.ini")
    command.revision(alembic_cfg, message=message, autogenerate=True)
    
    click.echo("Migration created successfully!")


@db.command("upgrade")
@click.option("--revision", default="head", help="Target revision")
def upgrade_db(revision: str):
    """Upgrade database to target revision."""
    from alembic.config import Config
    from alembic import command
    
    click.echo(f"Upgrading database to {revision}")
    
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, revision)
    
    click.echo("Database upgraded successfully!")


@db.command("downgrade")
@click.option("--revision", required=True, help="Target revision")
def downgrade_db(revision: str):
    """Downgrade database to target revision."""
    from alembic.config import Config
    from alembic import command
    
    click.echo(f"Downgrading database to {revision}")
    
    alembic_cfg = Config("alembic.ini")
    command.downgrade(alembic_cfg, revision)
    
    click.echo("Database downgraded successfully!")


@db.command("status")
def db_status():
    """Show database migration status."""
    from alembic.config import Config
    from alembic import command
    
    alembic_cfg = Config("alembic.ini")
    command.current(alembic_cfg)
    command.show(alembic_cfg, "head")


@cli.group()
def cache():
    """Cache management commands."""
    pass


@cache.command("clear")
@click.option("--pattern", help="Key pattern to clear (e.g., 'user:*')")
def clear_cache(pattern: Optional[str]):
    """Clear Redis cache."""
    async def clear():
        await redis_manager.initialize()
        
        if pattern:
            count = await redis_manager.flush_cache(pattern)
            click.echo(f"Cleared {count} keys matching pattern: {pattern}")
        else:
            await redis_manager.flush_cache()
            click.echo("Cleared entire cache")
        
        await redis_manager.close()
    
    asyncio.run(clear())


@cache.command("info")
def cache_info():
    """Show Redis cache information."""
    async def info():
        await redis_manager.initialize()
        
        info_data = await redis_manager.get_info()
        
        click.echo("Redis Information:")
        for key, value in info_data.items():
            click.echo(f"  {key}: {value}")
        
        await redis_manager.close()
    
    asyncio.run(info())


@cli.group()
def health():
    """Health check commands."""
    pass


@health.command("check")
def health_check():
    """Perform health check on all services."""
    async def check():
        click.echo("Performing health checks...")
        
        # Database health
        try:
            await db_manager.initialize()
            db_healthy = await db_manager.health_check()
            click.echo(f"Database: {'✓ Healthy' if db_healthy else '✗ Unhealthy'}")
            await db_manager.close()
        except Exception as e:
            click.echo(f"Database: ✗ Error - {e}")
            db_healthy = False
        
        # Redis health
        try:
            await redis_manager.initialize()
            redis_healthy = await redis_manager.health_check()
            click.echo(f"Redis: {'✓ Healthy' if redis_healthy else '✗ Unhealthy'}")
            await redis_manager.close()
        except Exception as e:
            click.echo(f"Redis: ✗ Error - {e}")
            redis_healthy = False
        
        # Overall status
        overall_healthy = db_healthy and redis_healthy
        click.echo(f"\nOverall Status: {'✓ All services healthy' if overall_healthy else '✗ Some services unhealthy'}")
        
        return 0 if overall_healthy else 1
    
    result = asyncio.run(check())
    exit(result)


@cli.command()
def shell():
    """Start interactive Python shell with app context."""
    import IPython
    from app.config import settings
    from app.models import User
    
    # Create namespace for shell
    namespace = {
        "settings": settings,
        "User": User,
        "db_manager": db_manager,
        "redis_manager": redis_manager,
    }
    
    click.echo("Starting interactive shell...")
    click.echo("Available objects: settings, User, db_manager, redis_manager")
    
    IPython.start_ipython(argv=[], user_ns=namespace)


@cli.command()
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]))
def status(output_format: str):
    """Show application status."""
    async def get_status():
        status_info = {
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
        }
        
        try:
            await db_manager.initialize()
            status_info["database"] = "healthy" if await db_manager.health_check() else "unhealthy"
            await db_manager.close()
        except Exception:
            status_info["database"] = "error"
        
        try:
            await redis_manager.initialize()
            status_info["redis"] = "healthy" if await redis_manager.health_check() else "unhealthy"
            await redis_manager.close()
        except Exception:
            status_info["redis"] = "error"
        
        return status_info
    
    status_info = asyncio.run(get_status())
    
    if output_format == "json":
        import json
        click.echo(json.dumps(status_info, indent=2))
    else:
        click.echo("Application Status:")
        for key, value in status_info.items():
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    cli()