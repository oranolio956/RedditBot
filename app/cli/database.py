"""
Database Management CLI

Command-line interface for database operations including initialization,
backup, maintenance, and monitoring.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich import print as rprint

from app.database.init_db import DatabaseInitializer
from app.database.manager import DatabaseService
from app.database.connection import db_manager
from app.database.repositories import RepositoryFactory
from app.config import settings

console = Console()


@click.group()
def db():
    """Database management commands."""
    pass


@db.command()
@click.option('--drop-existing', is_flag=True, help='Drop existing tables before initialization')
@click.option('--sample-data', is_flag=True, help='Create sample data for development')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def init(drop_existing: bool, sample_data: bool, verbose: bool):
    """Initialize database with tables, indexes, and sample data."""
    
    async def run_init():
        console.print("[bold green]🚀 Starting database initialization...[/bold green]")
        
        await db_manager.initialize()
        
        try:
            initializer = DatabaseInitializer()
            
            # Override sample data setting if specified
            if sample_data:
                initializer.sample_data_enabled = True
            
            with console.status("[bold green]Initializing database..."):
                results = await initializer.initialize_database(drop_existing=drop_existing)
            
            if results['success']:
                console.print("[bold green]✅ Database initialization completed successfully![/bold green]")
                
                # Display results table
                table = Table(title="Initialization Results")
                table.add_column("Step", style="cyan")
                table.add_column("Status", justify="center")
                table.add_column("Message", style="green")
                
                for step, result in results['steps'].items():
                    status = "✅ Success" if result.get('success', False) else "❌ Failed"
                    message = result.get('message', result.get('error', 'No details'))
                    table.add_row(step.replace('_', ' ').title(), status, message)
                
                console.print(table)
                
            else:
                console.print("[bold red]❌ Database initialization failed![/bold red]")
                
                for step, result in results['steps'].items():
                    if not result.get('success', False):
                        console.print(f"[red]❌ {step}: {result.get('error', 'Unknown error')}[/red]")
                
                if results.get('errors'):
                    for error in results['errors']:
                        console.print(f"[red]❌ {error}[/red]")
        
        finally:
            await db_manager.close()
    
    asyncio.run(run_init())


@db.command()
@click.option('--backup-type', default='manual', help='Backup type (manual, scheduled, full)')
@click.option('--output', '-o', help='Output directory for backup file')
def backup(backup_type: str, output: Optional[str]):
    """Create database backup."""
    
    async def run_backup():
        console.print(f"[bold blue]📁 Creating {backup_type} backup...[/bold blue]")
        
        db_service = DatabaseService()
        await db_service.initialize()
        
        try:
            with console.status("[bold blue]Creating backup..."):
                result = await db_service.create_backup(backup_type)
            
            if result['success']:
                console.print("[bold green]✅ Backup completed successfully![/bold green]")
                
                # Display backup info
                info_panel = Panel.fit(
                    f"[green]Backup File:[/green] {result['backup_file']}\n"
                    f"[green]Size:[/green] {result['size_mb']:.2f} MB\n"
                    f"[green]Duration:[/green] {result['duration_seconds']:.2f} seconds\n"
                    f"[green]Compressed:[/green] {result.get('compressed', False)}",
                    title="Backup Details"
                )
                console.print(info_panel)
                
            else:
                console.print(f"[bold red]❌ Backup failed: {result['error']}[/bold red]")
        
        finally:
            await db_service.shutdown()
    
    asyncio.run(run_backup())


@db.command()
@click.argument('backup_file')
@click.option('--target-db', help='Target database name (optional)')
@click.option('--confirm', is_flag=True, help='Confirm restore operation')
def restore(backup_file: str, target_db: Optional[str], confirm: bool):
    """Restore database from backup file."""
    
    if not confirm:
        console.print("[yellow]⚠️  This operation will restore database from backup.[/yellow]")
        console.print("[yellow]Use --confirm flag to proceed.[/yellow]")
        return
    
    async def run_restore():
        console.print(f"[bold blue]🔄 Restoring database from {backup_file}...[/bold blue]")
        
        db_service = DatabaseService()
        await db_service.initialize()
        
        try:
            with console.status("[bold blue]Restoring database..."):
                result = await db_service.restore_backup(backup_file, target_db)
            
            if result['success']:
                console.print("[bold green]✅ Restore completed successfully![/bold green]")
                console.print(f"[green]Target database:[/green] {result['target_database']}")
                console.print(f"[green]Duration:[/green] {result['duration_seconds']:.2f} seconds")
            else:
                console.print(f"[bold red]❌ Restore failed: {result['error']}[/bold red]")
        
        finally:
            await db_service.shutdown()
    
    asyncio.run(run_restore())


@db.command('list-backups')
def list_backups():
    """List available database backups."""
    
    async def run_list():
        db_service = DatabaseService()
        await db_service.initialize()
        
        try:
            backups = await db_service.list_backups()
            
            if not backups:
                console.print("[yellow]No backups found.[/yellow]")
                return
            
            table = Table(title="Available Backups")
            table.add_column("Filename", style="cyan")
            table.add_column("Size (MB)", justify="right", style="green")
            table.add_column("Created", style="blue")
            table.add_column("Compressed", justify="center")
            
            for backup in backups:
                created = datetime.fromisoformat(backup['created_at']).strftime("%Y-%m-%d %H:%M")
                compressed = "✅" if backup['compressed'] else "❌"
                table.add_row(
                    backup['filename'],
                    f"{backup['size_mb']:.2f}",
                    created,
                    compressed
                )
            
            console.print(table)
        
        finally:
            await db_service.shutdown()
    
    asyncio.run(run_list())


@db.command()
@click.option('--operations', help='Comma-separated list of operations (vacuum,analyze,reindex,cleanup_logs)')
def maintenance(operations: Optional[str]):
    """Perform database maintenance operations."""
    
    op_list = operations.split(',') if operations else None
    
    async def run_maintenance():
        console.print("[bold yellow]🔧 Performing database maintenance...[/bold yellow]")
        
        db_service = DatabaseService()
        await db_service.initialize()
        
        try:
            with console.status("[bold yellow]Running maintenance operations..."):
                result = await db_service.perform_maintenance(op_list)
            
            # Display results
            table = Table(title="Maintenance Results")
            table.add_column("Operation", style="cyan")
            table.add_column("Status", justify="center")
            table.add_column("Details", style="green")
            
            for operation, details in result['operations'].items():
                status = "✅ Success" if details.get('success', False) else "❌ Failed"
                message = details.get('message', details.get('error', 'No details'))
                table.add_row(operation.replace('_', ' ').title(), status, message)
            
            console.print(table)
            
            # Summary
            summary = result['summary']
            console.print(f"\n[green]✅ Successful operations: {summary['successful']}[/green]")
            if summary['failed'] > 0:
                console.print(f"[red]❌ Failed operations: {summary['failed']}[/red]")
        
        finally:
            await db_service.shutdown()
    
    asyncio.run(run_maintenance())


@db.command()
@click.option('--format', 'output_format', default='table', help='Output format (table, json)')
def health():
    """Check database health status."""
    
    async def run_health():
        console.print("[bold blue]🏥 Checking database health...[/bold blue]")
        
        db_service = DatabaseService()
        await db_service.initialize()
        
        try:
            with console.status("[bold blue]Running health checks..."):
                health_data = await db_service.get_health_status()
            
            if output_format == 'json':
                console.print(json.dumps(health_data, indent=2, default=str))
                return
            
            # Display overall status
            status_color = "green" if health_data['overall_status'] == 'healthy' else "red"
            console.print(f"\n[bold {status_color}]Overall Status: {health_data['overall_status'].upper()}[/bold {status_color}]")
            
            # Display checks
            if 'checks' in health_data:
                table = Table(title="Health Check Results")
                table.add_column("Check", style="cyan")
                table.add_column("Status", justify="center")
                table.add_column("Details")
                
                for check_name, check_result in health_data['checks'].items():
                    status = "✅ Healthy" if check_result.get('healthy', False) else "❌ Unhealthy"
                    
                    # Format details
                    details = []
                    for key, value in check_result.items():
                        if key not in ['healthy', 'error'] and value is not None:
                            if isinstance(value, (int, float)):
                                details.append(f"{key}: {value}")
                            elif isinstance(value, bool):
                                details.append(f"{key}: {'Yes' if value else 'No'}")
                    
                    if check_result.get('error'):
                        details.append(f"Error: {check_result['error']}")
                    
                    table.add_row(
                        check_name.replace('_', ' ').title(),
                        status,
                        ', '.join(details) if details else 'No details'
                    )
                
                console.print(table)
            
            # Display alerts if any
            if health_data.get('alerts'):
                console.print(f"\n[bold red]🚨 Alerts:[/bold red]")
                for alert in health_data['alerts']:
                    console.print(f"[red]• {alert}[/red]")
        
        finally:
            await db_service.shutdown()
    
    asyncio.run(run_health())


@db.command()
@click.option('--format', 'output_format', default='table', help='Output format (table, json)')
def stats():
    """Display database statistics."""
    
    async def run_stats():
        console.print("[bold blue]📊 Gathering database statistics...[/bold blue]")
        
        db_service = DatabaseService()
        await db_service.initialize()
        
        try:
            with console.status("[bold blue]Collecting statistics..."):
                stats_data = await db_service.get_statistics()
            
            if output_format == 'json':
                console.print(json.dumps(stats_data, indent=2, default=str))
                return
            
            # Database-wide stats
            if 'database_stats' in stats_data:
                db_stats = stats_data['database_stats']
                
                info_panel = Panel.fit(
                    f"[blue]Active Connections:[/blue] {db_stats.get('active_connections', 'N/A')}\n"
                    f"[blue]Transactions Committed:[/blue] {db_stats.get('transactions_committed', 'N/A'):,}\n"
                    f"[blue]Transactions Rolled Back:[/blue] {db_stats.get('transactions_rolled_back', 'N/A'):,}\n"
                    f"[blue]Cache Hit Ratio:[/blue] {stats_data.get('cache_hit_ratio_percent', 'N/A'):.2f}%\n"
                    f"[blue]Total Tables:[/blue] {stats_data.get('total_tables', 'N/A')}\n"
                    f"[blue]Total Live Tuples:[/blue] {stats_data.get('total_live_tuples', 'N/A'):,}\n"
                    f"[blue]Total Dead Tuples:[/blue] {stats_data.get('total_dead_tuples', 'N/A'):,}",
                    title="Database Statistics"
                )
                console.print(info_panel)
            
            # Table statistics
            if 'table_stats' in stats_data:
                table = Table(title="Table Statistics")
                table.add_column("Table", style="cyan")
                table.add_column("Live Tuples", justify="right", style="green")
                table.add_column("Dead Tuples", justify="right", style="red")
                table.add_column("Inserts", justify="right", style="blue")
                table.add_column("Updates", justify="right", style="yellow")
                table.add_column("Deletes", justify="right", style="red")
                
                for table_stat in stats_data['table_stats'][:10]:  # Show top 10 tables
                    table.add_row(
                        table_stat['tablename'],
                        f"{table_stat['live_tuples'] or 0:,}",
                        f"{table_stat['dead_tuples'] or 0:,}",
                        f"{table_stat['inserts'] or 0:,}",
                        f"{table_stat['updates'] or 0:,}",
                        f"{table_stat['deletes'] or 0:,}"
                    )
                
                console.print(table)
        
        finally:
            await db_service.shutdown()
    
    asyncio.run(run_stats())


@db.command()
@click.argument('repository_name')
@click.option('--limit', default=10, help='Number of records to show')
def inspect(repository_name: str, limit: int):
    """Inspect data in a specific repository/table."""
    
    async def run_inspect():
        await db_manager.initialize()
        
        try:
            # Get repository
            factory = RepositoryFactory()
            
            available_repos = list(factory._repositories.keys())
            
            if repository_name not in available_repos:
                console.print(f"[red]❌ Unknown repository: {repository_name}[/red]")
                console.print(f"[blue]Available repositories:[/blue] {', '.join(available_repos)}")
                return
            
            repo = factory.get_repository(repository_name)
            
            with console.status(f"[blue]Inspecting {repository_name}..."):
                # Get record count
                total_count = await repo.count_by_filters([])
                
                # Get recent records
                from app.database.repository import PaginationParams
                pagination = PaginationParams(page=1, size=limit)
                result = await repo.find_by_filters(pagination=pagination)
            
            console.print(f"\n[bold blue]Repository: {repository_name}[/bold blue]")
            console.print(f"[blue]Total records: {total_count:,}[/blue]")
            console.print(f"[blue]Showing: {len(result.items)} records[/blue]")
            
            if result.items:
                # Display first record as example
                sample_record = result.items[0].to_dict()
                
                table = Table(title=f"Sample Records from {repository_name}")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="green")
                
                for key, value in sample_record.items():
                    # Truncate long values
                    str_value = str(value)
                    if len(str_value) > 100:
                        str_value = str_value[:100] + "..."
                    
                    table.add_row(key, str_value)
                
                console.print(table)
                
                if len(result.items) > 1:
                    console.print(f"\n[yellow]... and {len(result.items) - 1} more records[/yellow]")
            else:
                console.print("[yellow]No records found.[/yellow]")
        
        finally:
            await db_manager.close()
    
    asyncio.run(run_inspect())


@db.command()
@click.option('--auto-yes', is_flag=True, help='Skip confirmation prompts')
def reset(auto_yes: bool):
    """Reset database (drop all tables and reinitialize)."""
    
    if not auto_yes:
        console.print("[bold red]⚠️  WARNING: This will delete ALL data in the database![/bold red]")
        if not click.confirm("Are you sure you want to continue?"):
            console.print("Operation cancelled.")
            return
    
    async def run_reset():
        console.print("[bold red]🔄 Resetting database...[/bold red]")
        
        await db_manager.initialize()
        
        try:
            initializer = DatabaseInitializer()
            
            with console.status("[bold red]Dropping tables and reinitializing..."):
                results = await initializer.initialize_database(drop_existing=True)
            
            if results['success']:
                console.print("[bold green]✅ Database reset completed successfully![/bold green]")
            else:
                console.print("[bold red]❌ Database reset failed![/bold red]")
                for step, result in results['steps'].items():
                    if not result.get('success', False):
                        console.print(f"[red]❌ {step}: {result.get('error', 'Unknown error')}[/red]")
        
        finally:
            await db_manager.close()
    
    asyncio.run(run_reset())


@db.command()
def config():
    """Show database configuration."""
    
    config_info = {
        "Database URL": f"postgresql://{settings.database.user}@{settings.database.host}:{settings.database.port}/{settings.database.name}",
        "Pool Size": settings.database.pool_size,
        "Max Overflow": settings.database.max_overflow,
        "Pool Timeout": f"{settings.database.pool_timeout}s",
        "Pool Recycle": f"{settings.database.pool_recycle}s",
        "Environment": settings.environment,
        "Debug Mode": settings.debug
    }
    
    table = Table(title="Database Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config_info.items():
        table.add_row(key, str(value))
    
    console.print(table)


if __name__ == "__main__":
    db()