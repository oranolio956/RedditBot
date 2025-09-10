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


@cli.group()
def engagement():
    """Proactive engagement system commands."""
    pass


@engagement.command("seed-milestones")
@click.option("--overwrite", is_flag=True, help="Overwrite existing milestones")
def seed_milestones(overwrite: bool):
    """Seed database with default engagement milestones."""
    async def seed():
        from app.services.milestone_seeder import seed_default_milestones
        
        await db_manager.initialize()
        
        try:
            results = await seed_default_milestones(overwrite_existing=overwrite)
            
            click.echo(f"✅ Milestones seeded successfully:")
            click.echo(f"   Created: {results['milestones_created']}")
            click.echo(f"   Updated: {results['milestones_updated']}")
            click.echo(f"   Skipped: {results['milestones_skipped']}")
            click.echo(f"   Total: {results['total_milestones']}")
            
        except Exception as e:
            click.echo(f"❌ Error seeding milestones: {e}")
        finally:
            await db_manager.close()
    
    asyncio.run(seed())


@engagement.command("analyze-patterns")
@click.option("--user-id", help="Analyze specific user by UUID")
@click.option("--telegram-id", type=int, help="Analyze specific user by Telegram ID")
@click.option("--batch-size", default=50, help="Number of users to analyze")
def analyze_patterns(user_id: Optional[str], telegram_id: Optional[int], batch_size: int):
    """Analyze user behavioral patterns."""
    async def analyze():
        from app.services.engagement_analyzer import EngagementAnalyzer
        
        await db_manager.initialize()
        analyzer = EngagementAnalyzer()
        
        try:
            if user_id or telegram_id:
                # Analyze specific user
                if telegram_id and not user_id:
                    # Get user_id from telegram_id
                    from app.database.connection import get_database_session
                    async with get_database_session() as session:
                        result = await session.execute(
                            "SELECT id FROM users WHERE telegram_id = :telegram_id",
                            {'telegram_id': telegram_id}
                        )
                        row = result.fetchone()
                        if row:
                            user_id = str(row.id)
                        else:
                            click.echo(f"❌ User with Telegram ID {telegram_id} not found")
                            return
                
                pattern = await analyzer.update_user_behavior_patterns(user_id, telegram_id or 0)
                
                if pattern:
                    click.echo(f"✅ Pattern analysis completed for user {user_id}:")
                    click.echo(f"   Churn Risk: {pattern.churn_risk_score:.3f}")
                    click.echo(f"   Total Interactions: {pattern.total_interactions}")
                    click.echo(f"   Needs Re-engagement: {pattern.needs_re_engagement}")
                    click.echo(f"   Engagement Trend: {pattern.engagement_quality_trend or 0:.3f}")
                else:
                    click.echo(f"❌ No pattern data available for user {user_id}")
            else:
                # Find users needing engagement
                candidates = await analyzer.find_users_needing_engagement(batch_size)
                
                click.echo(f"✅ Found {len(candidates)} users needing engagement:")
                for candidate in candidates[:10]:  # Show top 10
                    patterns = candidate['patterns']
                    click.echo(f"   {candidate['user']['display_name']}: "
                             f"Churn Risk {patterns['churn_risk_score']:.3f}, "
                             f"Days Since Last: {patterns['days_since_last_interaction']}")
                
                if len(candidates) > 10:
                    click.echo(f"   ... and {len(candidates) - 10} more")
                    
        except Exception as e:
            click.echo(f"❌ Error analyzing patterns: {e}")
        finally:
            await db_manager.close()
    
    asyncio.run(analyze())


@engagement.command("schedule-outreach")
@click.option("--user-id", required=True, help="User UUID")
@click.option("--type", "outreach_type", 
              type=click.Choice(['milestone_celebration', 're_engagement', 'personalized_checkin', 
                               'feature_suggestion', 'mood_support', 'topic_follow_up']),
              required=True, help="Type of outreach")
@click.option("--priority", default=0.5, help="Priority score (0-1)")
@click.option("--context", help="JSON context data")
def schedule_outreach(user_id: str, outreach_type: str, priority: float, context: Optional[str]):
    """Schedule a proactive outreach for a user."""
    async def schedule():
        import json
        from app.services.proactive_outreach import ProactiveOutreachService
        from app.models.engagement import OutreachType
        
        await db_manager.initialize()
        
        try:
            context_data = json.loads(context) if context else {}
            
            service = ProactiveOutreachService()
            outreach = await service.schedule_proactive_outreach(
                user_id=user_id,
                outreach_type=OutreachType(outreach_type),
                priority_score=priority,
                context_data=context_data
            )
            
            if outreach:
                click.echo(f"✅ Outreach scheduled:")
                click.echo(f"   Outreach ID: {outreach.id}")
                click.echo(f"   Type: {outreach.outreach_type.value}")
                click.echo(f"   Scheduled for: {outreach.scheduled_for}")
                click.echo(f"   Priority: {outreach.priority_score}")
            else:
                click.echo(f"❌ Failed to schedule outreach (user may have recent outreaches)")
                
        except Exception as e:
            click.echo(f"❌ Error scheduling outreach: {e}")
        finally:
            await db_manager.close()
    
    asyncio.run(schedule())


@engagement.command("process-outreaches")
@click.option("--limit", default=50, help="Maximum outreaches to process")
def process_outreaches(limit: int):
    """Process scheduled outreaches."""
    async def process():
        from app.services.proactive_outreach import ProactiveOutreachService
        
        await db_manager.initialize()
        service = ProactiveOutreachService()
        
        try:
            results = await service.process_scheduled_outreaches(limit)
            
            click.echo(f"✅ Processed {len(results)} outreaches:")
            successful = len([r for r in results if r['status'] == 'sent'])
            failed = len([r for r in results if r['status'] == 'failed'])
            
            click.echo(f"   Successful: {successful}")
            click.echo(f"   Failed: {failed}")
            
            if failed > 0:
                click.echo("\n❌ Failed outreaches:")
                for result in results:
                    if result['status'] == 'failed':
                        click.echo(f"   {result['outreach_id']}: {result.get('error', 'Unknown error')}")
                        
        except Exception as e:
            click.echo(f"❌ Error processing outreaches: {e}")
        finally:
            await db_manager.close()
    
    asyncio.run(process())


@engagement.command("analytics")
@click.option("--days", default=30, help="Number of days to analyze")
def get_analytics(days: int):
    """Get engagement analytics."""
    async def analyze():
        from app.services.proactive_outreach import ProactiveOutreachService
        from app.services.milestone_seeder import MilestoneSeeder
        
        await db_manager.initialize()
        
        try:
            # Outreach analytics
            service = ProactiveOutreachService()
            analytics = await service.get_outreach_analytics(days)
            
            click.echo(f"📊 Engagement Analytics ({days} days):")
            click.echo(f"\n📤 Outreach Performance:")
            click.echo(f"   Total Outreaches: {analytics['overall_metrics']['total_outreaches']}")
            click.echo(f"   Send Rate: {analytics['overall_metrics']['send_rate']:.1f}%")
            click.echo(f"   Response Rate: {analytics['overall_metrics']['response_rate']:.1f}%")
            
            click.echo(f"\n📈 By Outreach Type:")
            for type_data in analytics['by_outreach_type']:
                click.echo(f"   {type_data['type']}: {type_data['response_rate']:.1f}% "
                         f"({type_data['responses']}/{type_data['total_sent']})")
            
            # Milestone statistics
            seeder = MilestoneSeeder()
            milestone_stats = await seeder.get_milestone_stats()
            
            click.echo(f"\n🏆 Milestone Statistics:")
            click.echo(f"   Active Milestones: {milestone_stats['active_milestones']}")
            click.echo(f"   Total Milestones: {milestone_stats['total_milestones']}")
            
            if milestone_stats['achievement_stats']:
                click.echo(f"\n🎯 Top Achievements:")
                for stat in milestone_stats['achievement_stats'][:5]:
                    click.echo(f"   {stat['milestone_name']}: {stat['total_achievements']} achievements")
                    
        except Exception as e:
            click.echo(f"❌ Error getting analytics: {e}")
        finally:
            await db_manager.close()
    
    asyncio.run(analyze())


@engagement.command("train-models")
@click.option("--min-samples", default=500, help="Minimum samples for training")
def train_models(min_samples: int):
    """Train behavioral prediction models."""
    async def train():
        from app.services.behavioral_predictor import BehavioralPredictor
        
        await db_manager.initialize()
        predictor = BehavioralPredictor()
        
        try:
            click.echo(f"🧠 Training behavioral models (min samples: {min_samples})...")
            results = await predictor.train_models(min_samples)
            
            if 'error' in results:
                click.echo(f"❌ Training failed: {results['error']}")
            else:
                click.echo(f"✅ Training completed:")
                click.echo(f"   Models trained: {', '.join(results.get('models_trained', []))}")
                click.echo(f"   Training samples: {results.get('training_samples', 0)}")
                click.echo(f"   Model version: {results.get('model_version', 'unknown')}")
                
                # Show model performance metrics
                model_results = results.get('results', {})
                for model_name, metrics in model_results.items():
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        click.echo(f"   {model_name} accuracy: {metrics['accuracy']:.3f}")
                        
        except Exception as e:
            click.echo(f"❌ Error training models: {e}")
        finally:
            await db_manager.close()
    
    asyncio.run(train())


@engagement.command("user-insights")
@click.option("--telegram-id", type=int, required=True, help="Telegram user ID")
def user_insights(telegram_id: int):
    """Get engagement insights for a specific user."""
    async def get_insights():
        from app.services.engagement_integration_example import EngagementIntegration
        from app.database.connection import get_database_session
        
        await db_manager.initialize()
        
        try:
            # Get user
            async with get_database_session() as session:
                result = await session.execute(
                    "SELECT id, first_name, username, telegram_id FROM users WHERE telegram_id = :telegram_id",
                    {'telegram_id': telegram_id}
                )
                user_data = result.fetchone()
                
                if not user_data:
                    click.echo(f"❌ User with Telegram ID {telegram_id} not found")
                    return
                
                # Mock user object with required attributes
                class MockUser:
                    def __init__(self, data):
                        self.id = data.id
                        self.telegram_id = data.telegram_id
                        self.first_name = data.first_name
                        self.username = data.username
                    
                    def get_display_name(self):
                        return self.first_name or self.username or f"User {self.telegram_id}"
                
                user = MockUser(user_data)
                
                integration = EngagementIntegration()
                insights = await integration.get_user_engagement_insights(user)
                
                if 'error' in insights:
                    click.echo(f"❌ Error getting insights: {insights['error']}")
                    return
                
                click.echo(f"👤 User Insights: {user.get_display_name()} ({telegram_id})")
                
                patterns = insights['behavior_pattern']
                click.echo(f"\n📊 Behavior Patterns:")
                click.echo(f"   Total Interactions: {patterns['total_interactions']}")
                click.echo(f"   Churn Risk Score: {patterns['churn_risk_score']:.3f}")
                click.echo(f"   Engagement Trend: {patterns['engagement_trend']:.3f}")
                click.echo(f"   Days Since Last: {patterns['days_since_last_interaction']}")
                click.echo(f"   Needs Re-engagement: {patterns['needs_re_engagement']}")
                
                predictions = insights['predictions']
                click.echo(f"\n🔮 Predictions:")
                click.echo(f"   Churn Risk: {predictions['churn_risk']:.3f} (confidence: {predictions['churn_confidence']:.3f})")
                click.echo(f"   Mood Trend: {predictions['mood_trend']:.3f}")
                click.echo(f"   Optimal Outreach Hour: {predictions['optimal_outreach_hour']}")
                
                milestones = insights['milestones']
                if milestones:
                    click.echo(f"\n🏆 Milestones Progress:")
                    for milestone in milestones[:5]:
                        status = "✅" if milestone['achieved'] else f"{milestone['progress']:.1f}%"
                        click.echo(f"   {milestone['name']}: {status}")
                        
        except Exception as e:
            click.echo(f"❌ Error getting user insights: {e}")
        finally:
            await db_manager.close()
    
    asyncio.run(get_insights())


@engagement.command("start-tasks")
@click.option("--workers", default=2, help="Number of Celery workers")
@click.option("--loglevel", default="info", help="Log level")
def start_engagement_tasks(workers: int, loglevel: str):
    """Start Celery workers for engagement tasks."""
    import subprocess
    import time
    import signal
    import sys
    
    processes = []
    
    def signal_handler(sig, frame):
        click.echo("\n🛑 Stopping engagement task workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start Celery worker
        worker_cmd = [
            "celery", "-A", "app.services.engagement_tasks",
            "worker", 
            "--loglevel", loglevel,
            "--concurrency", str(workers)
        ]
        
        click.echo(f"🚀 Starting Celery worker with {workers} processes...")
        worker_process = subprocess.Popen(worker_cmd)
        processes.append(worker_process)
        
        # Start Celery beat scheduler
        beat_cmd = [
            "celery", "-A", "app.services.engagement_tasks",
            "beat",
            "--loglevel", loglevel
        ]
        
        click.echo(f"⏰ Starting Celery beat scheduler...")
        beat_process = subprocess.Popen(beat_cmd)
        processes.append(beat_process)
        
        click.echo(f"✅ Engagement task system started!")
        click.echo(f"   - Worker: PID {worker_process.pid}")
        click.echo(f"   - Beat: PID {beat_process.pid}")
        click.echo(f"   Press Ctrl+C to stop")
        
        # Wait for processes
        while True:
            time.sleep(1)
            for p in processes[:]:
                if p.poll() is not None:
                    click.echo(f"❌ Process {p.pid} died with code {p.returncode}")
                    processes.remove(p)
            
            if not processes:
                break
                
    except Exception as e:
        click.echo(f"❌ Error starting engagement tasks: {e}")
        for p in processes:
            p.terminate()


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