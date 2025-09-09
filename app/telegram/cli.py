"""
Telegram Bot CLI Management

Command-line interface for managing the Telegram bot with comprehensive
controls for webhook setup, monitoring, and maintenance.
"""

import asyncio
import sys
import json
from typing import Optional, Dict, Any
from pathlib import Path

import click
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
import aiofiles

from app.config import settings
from app.telegram.bot import TelegramBot

logger = structlog.get_logger(__name__)
console = Console()


class TelegramBotCLI:
    """CLI for Telegram bot management."""
    
    def __init__(self):
        self.bot: Optional[TelegramBot] = None
        self.console = Console()
    
    async def initialize_bot(self) -> None:
        """Initialize bot instance."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task("Initializing bot...", total=None)
                
                self.bot = TelegramBot()
                await self.bot.initialize()
                
                progress.update(task, description="Bot initialized successfully ✓")
            
        except Exception as e:
            self.console.print(f"[red]Failed to initialize bot: {e}[/red]")
            raise
    
    async def cleanup_bot(self) -> None:
        """Cleanup bot resources."""
        if self.bot:
            await self.bot.cleanup()
            self.bot = None
    
    def display_status(self, status: Dict[str, Any]) -> None:
        """Display bot status in a formatted table."""
        table = Table(title="Telegram Bot Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # Basic status
        table.add_row(
            "Bot",
            "🟢 Running" if status['is_running'] else "🔴 Stopped",
            f"Uptime: {status.get('uptime', 0):.1f}s"
        )
        
        table.add_row(
            "Messages",
            str(status['message_count']),
            f"Errors: {status['error_count']}"
        )
        
        table.add_row(
            "Connections",
            str(status['current_connections']),
            f"Queue: {status['queue_size']} messages"
        )
        
        # Component status
        components = status.get('components', {})
        for component, active in components.items():
            table.add_row(
                component.replace('_', ' ').title(),
                "🟢 Active" if active else "🔴 Inactive",
                ""
            )
        
        # Health status
        health = status.get('health', {})
        for service, healthy in health.items():
            table.add_row(
                f"Health: {service}",
                "🟢 Healthy" if healthy else "🔴 Unhealthy",
                ""
            )
        
        self.console.print(table)
    
    def display_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display bot metrics."""
        # Performance metrics
        perf_table = Table(title="Performance Metrics", show_header=True)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        perf_table.add_column("Unit", style="dim")
        
        performance = metrics.get('performance', {})
        perf_table.add_row("CPU Usage", f"{performance.get('cpu_usage', 0):.1f}", "%")
        perf_table.add_row("Memory Usage", f"{performance.get('memory_usage', 0):.1f}", "%")
        perf_table.add_row("Avg Response Time", f"{performance.get('avg_response_time', 0):.3f}", "seconds")
        perf_table.add_row("Throughput", str(performance.get('recent_throughput_per_minute', 0)), "msg/min")
        perf_table.add_row("Concurrent Connections", str(performance.get('concurrent_connections', 0)), "connections")
        
        # Behavior metrics
        behavior_table = Table(title="User Behavior", show_header=True)
        behavior_table.add_column("Metric", style="cyan")
        behavior_table.add_column("Value", style="green")
        
        behavior = metrics.get('behavior', {})
        behavior_table.add_row("Active Users", str(behavior.get('active_users', 0)))
        behavior_table.add_row("Total Messages", str(behavior.get('total_messages', 0)))
        behavior_table.add_row("Total Errors", str(behavior.get('total_errors', 0)))
        
        # Anti-detection metrics
        anti_detection_table = Table(title="Anti-Detection System", show_header=True)
        anti_detection_table.add_column("Metric", style="cyan")
        anti_detection_table.add_column("Value", style="green")
        
        anti_detection = metrics.get('anti_detection', {})
        anti_detection_table.add_row("Risk Mitigations", str(anti_detection.get('risk_mitigations', 0)))
        anti_detection_table.add_row("Pattern Variations", str(anti_detection.get('patterns_applied', 0)))
        anti_detection_table.add_row("Detection Accuracy", f"{anti_detection.get('detection_accuracy', 0):.2%}")
        
        self.console.print(perf_table)
        self.console.print()
        self.console.print(behavior_table)
        self.console.print()
        self.console.print(anti_detection_table)
    
    def display_webhook_info(self, webhook_info: Dict[str, Any]) -> None:
        """Display webhook information."""
        panel_content = []
        
        if webhook_info.get('url'):
            panel_content.append(f"🔗 URL: {webhook_info['url']}")
        else:
            panel_content.append("❌ No webhook configured")
        
        panel_content.append(f"📋 Pending Updates: {webhook_info.get('pending_update_count', 0)}")
        
        if webhook_info.get('has_custom_certificate'):
            panel_content.append("🔒 Custom Certificate: Yes")
        
        if webhook_info.get('last_error_message'):
            panel_content.append(f"❌ Last Error: {webhook_info['last_error_message']}")
        
        panel_content.append(f"🔢 Max Connections: {webhook_info.get('max_connections', 'N/A')}")
        
        self.console.print(Panel(
            "\n".join(panel_content),
            title="Webhook Information",
            border_style="blue"
        ))


@click.group()
@click.pass_context
def cli(ctx):
    """Telegram Bot Management CLI."""
    ctx.ensure_object(dict)
    ctx.obj['cli'] = TelegramBotCLI()


@cli.command()
@click.pass_context
async def start(ctx):
    """Start the bot in polling mode."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        console.print("[green]Bot initialized successfully![/green]")
        
        console.print("[yellow]Starting bot polling...[/yellow]")
        await cli_instance.bot.start_polling()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down bot...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        if cli_instance.bot:
            await cli_instance.cleanup_bot()
        console.print("[green]Bot shutdown complete.[/green]")


@cli.command()
@click.pass_context
async def status(ctx):
    """Show bot status."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching status...", total=None)
            
            status_data = await cli_instance.bot.get_status()
            progress.update(task, description="Status retrieved ✓")
        
        cli_instance.display_status(status_data)
        
    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")
    finally:
        await cli_instance.cleanup_bot()


@cli.command()
@click.pass_context
async def metrics(ctx):
    """Show bot metrics."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching metrics...", total=None)
            
            metrics_data = await cli_instance.bot.metrics.get_current_metrics()
            progress.update(task, description="Metrics retrieved ✓")
        
        cli_instance.display_metrics(metrics_data)
        
    except Exception as e:
        console.print(f"[red]Error getting metrics: {e}[/red]")
    finally:
        await cli_instance.cleanup_bot()


@cli.group()
def webhook():
    """Webhook management commands."""
    pass


@webhook.command('info')
@click.pass_context
async def webhook_info(ctx):
    """Show webhook information."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        
        if not cli_instance.bot.webhook_manager:
            console.print("[red]Webhook not configured[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching webhook info...", total=None)
            
            webhook_data = await cli_instance.bot.webhook_manager.get_webhook_info()
            progress.update(task, description="Webhook info retrieved ✓")
        
        cli_instance.display_webhook_info(webhook_data)
        
    except Exception as e:
        console.print(f"[red]Error getting webhook info: {e}[/red]")
    finally:
        await cli_instance.cleanup_bot()


@webhook.command('test')
@click.pass_context
async def webhook_test(ctx):
    """Test webhook connectivity."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        
        if not cli_instance.bot.webhook_manager:
            console.print("[red]Webhook not configured[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Testing webhook...", total=None)
            
            test_result = await cli_instance.bot.webhook_manager.test_webhook_connectivity()
            progress.update(task, description="Webhook test complete ✓")
        
        # Display test results
        if test_result.get('status') == 'completed':
            console.print("[green]✓ Webhook test completed[/green]")
            
            connectivity = test_result.get('connectivity', {})
            if connectivity.get('url_accessible'):
                console.print(f"[green]✓ URL accessible (status: {connectivity.get('status_code')})[/green]")
            else:
                console.print(f"[red]✗ URL not accessible: {connectivity.get('error')}[/red]")
            
            telegram_info = test_result.get('telegram_webhook_info', {})
            if telegram_info.get('url'):
                console.print(f"[green]✓ Telegram webhook configured[/green]")
            else:
                console.print("[yellow]⚠ No webhook configured in Telegram[/yellow]")
        else:
            console.print(f"[red]✗ Webhook test failed: {test_result.get('error')}[/red]")
        
    except Exception as e:
        console.print(f"[red]Error testing webhook: {e}[/red]")
    finally:
        await cli_instance.cleanup_bot()


@cli.command()
@click.option('--user-id', type=int, help='User ID to check rate limits for')
@click.pass_context
async def rate_limits(ctx, user_id: Optional[int]):
    """Show rate limit status."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        
        if not cli_instance.bot.rate_limiter:
            console.print("[red]Rate limiter not available[/red]")
            return
        
        if user_id:
            # Show user-specific rate limits
            status_data = await cli_instance.bot.rate_limiter.get_status(str(user_id))
            
            table = Table(title=f"Rate Limits for User {user_id}", show_header=True)
            table.add_column("Rule", style="cyan")
            table.add_column("Current", style="yellow")
            table.add_column("Limit", style="green")
            table.add_column("Remaining", style="blue")
            table.add_column("Limited", style="red")
            
            for rule_name, rule_data in status_data.items():
                table.add_row(
                    rule_name,
                    str(rule_data.get('current_requests', 0)),
                    str(rule_data.get('limit', 0)),
                    str(rule_data.get('remaining', 0)),
                    "Yes" if rule_data.get('is_limited', False) else "No"
                )
            
            console.print(table)
        else:
            # Show general rate limiter metrics
            metrics = await cli_instance.bot.rate_limiter.get_metrics()
            
            table = Table(title="Rate Limiter Metrics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Rules", str(metrics.get('total_rules', 0)))
            table.add_row("Active Limits", str(metrics.get('active_limits', 0)))
            table.add_row("Cache Size", str(metrics.get('local_cache_size', 0)))
            
            console.print(table)
            
            # Show rules
            rules_tree = Tree("Rate Limit Rules")
            rules = metrics.get('rules', {})
            
            for rule_name, rule_config in rules.items():
                rule_node = rules_tree.add(f"[cyan]{rule_name}[/cyan]")
                rule_node.add(f"Limit: {rule_config['limit']}")
                rule_node.add(f"Window: {rule_config['window']}s")
                rule_node.add(f"Priority: {rule_config['priority']}")
                if rule_config['burst_limit'] > 0:
                    rule_node.add(f"Burst: {rule_config['burst_limit']}")
            
            console.print(rules_tree)
        
    except Exception as e:
        console.print(f"[red]Error getting rate limits: {e}[/red]")
    finally:
        await cli_instance.cleanup_bot()


@cli.command()
@click.pass_context
async def sessions(ctx):
    """Show active sessions."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        
        if not cli_instance.bot.session_manager:
            console.print("[red]Session manager not available[/red]")
            return
        
        metrics = await cli_instance.bot.session_manager.get_metrics()
        
        # Summary table
        summary_table = Table(title="Session Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Active Sessions", str(metrics.get('active_sessions', 0)))
        summary_table.add_row("Total Users", str(metrics.get('total_users', 0)))
        summary_table.add_row("Sessions Created", str(metrics.get('total_sessions_created', 0)))
        summary_table.add_row("Sessions Expired", str(metrics.get('total_sessions_expired', 0)))
        summary_table.add_row("Avg Duration", f"{metrics.get('average_session_duration', 0):.1f}s")
        summary_table.add_row("Avg Interactions", f"{metrics.get('average_interactions_per_session', 0):.1f}")
        
        console.print(summary_table)
        
        # State distribution
        state_dist = metrics.get('state_distribution', {})
        if state_dist:
            console.print("\n[bold]Session States:[/bold]")
            for state, count in state_dist.items():
                console.print(f"  {state}: {count}")
        
    except Exception as e:
        console.print(f"[red]Error getting sessions: {e}[/red]")
    finally:
        await cli_instance.cleanup_bot()


@cli.command()
@click.pass_context
async def cleanup(ctx):
    """Run maintenance cleanup."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Session cleanup
            if cli_instance.bot.session_manager:
                task = progress.add_task("Cleaning expired sessions...", total=None)
                cleaned_sessions = await cli_instance.bot.session_manager.cleanup_expired_sessions()
                progress.update(task, description=f"Cleaned {cleaned_sessions} expired sessions ✓")
            
            # Rate limiter cleanup
            if cli_instance.bot.rate_limiter:
                task = progress.add_task("Cleaning rate limit data...", total=None)
                await cli_instance.bot.rate_limiter.cleanup_expired_data()
                progress.update(task, description="Rate limit data cleaned ✓")
            
            # Anti-ban cleanup
            if cli_instance.bot.anti_ban:
                task = progress.add_task("Cleaning anti-ban data...", total=None)
                await cli_instance.bot.anti_ban.cleanup_old_data()
                progress.update(task, description="Anti-ban data cleaned ✓")
        
        console.print("[green]✓ Cleanup completed successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during cleanup: {e}[/red]")
    finally:
        await cli_instance.cleanup_bot()


@cli.command()
@click.option('--format', type=click.Choice(['json', 'yaml']), default='json', help='Output format')
@click.option('--output', type=click.Path(), help='Output file path')
@click.pass_context
async def export_config(ctx, format: str, output: Optional[str]):
    """Export current configuration."""
    cli_instance = ctx.obj['cli']
    
    try:
        await cli_instance.initialize_bot()
        
        config_data = {
            'telegram': {
                'bot_token': settings.telegram.bot_token[:10] + '...',  # Masked
                'webhook_url': settings.telegram.webhook_url,
                'rate_limit_calls': settings.telegram.rate_limit_calls,
                'rate_limit_period': settings.telegram.rate_limit_period,
            },
            'redis': {
                'host': settings.redis.host,
                'port': settings.redis.port,
                'db': settings.redis.db,
                'max_connections': settings.redis.max_connections,
            },
            'monitoring': {
                'log_level': settings.monitoring.log_level,
                'metrics_enabled': settings.monitoring.metrics_enabled,
            }
        }
        
        if format == 'json':
            config_str = json.dumps(config_data, indent=2)
        else:  # yaml
            import yaml
            config_str = yaml.dump(config_data, default_flow_style=False)
        
        if output:
            async with aiofiles.open(output, 'w') as f:
                await f.write(config_str)
            console.print(f"[green]✓ Configuration exported to {output}[/green]")
        else:
            console.print(config_str)
        
    except Exception as e:
        console.print(f"[red]Error exporting config: {e}[/red]")
    finally:
        await cli_instance.cleanup_bot()


def run_async_command(coro):
    """Run async command in event loop."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context
            return loop.create_task(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(coro)


# Make all async commands work with click
for command in [start, status, metrics, webhook_info, webhook_test, rate_limits, sessions, cleanup, export_config]:
    command.callback = lambda ctx, *args, **kwargs, cmd=command.callback: run_async_command(cmd(ctx, *args, **kwargs))


if __name__ == '__main__':
    cli()