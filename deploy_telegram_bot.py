#!/usr/bin/env python3
"""
Production Deployment Script for Telegram Bot

Handles deployment, scaling, monitoring setup, and health checks
for the production Telegram bot infrastructure.
"""

import asyncio
import os
import sys
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import subprocess
import signal

import click
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
import docker
import psutil
import yaml

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
console = Console()


class TelegramBotDeployer:
    """Production deployment manager for Telegram bot."""
    
    def __init__(self):
        self.console = Console()
        self.docker_client = None
        self.project_root = Path(__file__).parent
        
        # Deployment configuration
        self.config = {
            'app_name': 'telegram-ml-bot',
            'version': '1.0.0',
            'environment': 'production',
            'replicas': 3,
            'memory_limit': '2G',
            'cpu_limit': '1.0',
            'health_check_interval': 30,
            'restart_policy': 'unless-stopped',
            'networks': ['telegram-bot-network'],
            'volumes': [
                'telegram-bot-data:/app/data',
                'telegram-bot-logs:/app/logs',
                'telegram-bot-models:/app/models'
            ]
        }
        
        # Service dependencies
        self.services = {
            'redis': {
                'image': 'redis:7-alpine',
                'port': 6379,
                'config': {
                    'maxmemory': '512mb',
                    'maxmemory-policy': 'allkeys-lru'
                }
            },
            'postgresql': {
                'image': 'postgres:15-alpine',
                'port': 5432,
                'config': {
                    'max_connections': 200,
                    'shared_buffers': '256MB'
                }
            },
            'prometheus': {
                'image': 'prom/prometheus:latest',
                'port': 9090,
                'config_path': './monitoring/prometheus.yml'
            },
            'grafana': {
                'image': 'grafana/grafana:latest',
                'port': 3000,
                'dashboards': './monitoring/dashboards/'
            }
        }
    
    async def initialize_docker(self) -> None:
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Docker client", error=str(e))
            raise
    
    def create_docker_compose(self) -> None:
        """Generate docker-compose.yml for production deployment."""
        compose_config = {
            'version': '3.8',
            'services': {},
            'networks': {
                'telegram-bot-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'telegram-bot-data': {},
                'telegram-bot-logs': {},
                'telegram-bot-models': {},
                'postgres-data': {},
                'redis-data': {},
                'prometheus-data': {},
                'grafana-data': {}
            }
        }
        
        # Main application service
        compose_config['services']['telegram-bot'] = {
            'build': {
                'context': '.',
                'dockerfile': 'Dockerfile'
            },
            'image': f"{self.config['app_name']}:{self.config['version']}",
            'container_name': f"{self.config['app_name']}-app",
            'restart': self.config['restart_policy'],
            'environment': [
                'ENVIRONMENT=production',
                'DEBUG=false',
                'DB_HOST=postgresql',
                'REDIS_HOST=redis',
                'TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}',
                'TELEGRAM_WEBHOOK_URL=${TELEGRAM_WEBHOOK_URL}',
                'TELEGRAM_WEBHOOK_SECRET=${TELEGRAM_WEBHOOK_SECRET}',
                'SECRET_KEY=${SECRET_KEY}',
                'JWT_SECRET_KEY=${JWT_SECRET_KEY}',
                'DB_PASSWORD=${DB_PASSWORD}',
                'REDIS_PASSWORD=${REDIS_PASSWORD}'
            ],
            'ports': ['8000:8000'],
            'volumes': [
                'telegram-bot-data:/app/data',
                'telegram-bot-logs:/app/logs',
                'telegram-bot-models:/app/models'
            ],
            'networks': ['telegram-bot-network'],
            'depends_on': ['postgresql', 'redis'],
            'deploy': {
                'replicas': self.config['replicas'],
                'resources': {
                    'limits': {
                        'memory': self.config['memory_limit'],
                        'cpus': self.config['cpu_limit']
                    }
                }
            },
            'healthcheck': {
                'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                'interval': f"{self.config['health_check_interval']}s",
                'timeout': '10s',
                'retries': 3,
                'start_period': '60s'
            }
        }
        
        # Redis service
        compose_config['services']['redis'] = {
            'image': self.services['redis']['image'],
            'container_name': f"{self.config['app_name']}-redis",
            'restart': 'unless-stopped',
            'ports': ['6379:6379'],
            'volumes': ['redis-data:/data'],
            'networks': ['telegram-bot-network'],
            'command': [
                'redis-server',
                '--maxmemory', self.services['redis']['config']['maxmemory'],
                '--maxmemory-policy', self.services['redis']['config']['maxmemory-policy'],
                '--save', '60', '1000',
                '--appendonly', 'yes'
            ],
            'healthcheck': {
                'test': ['CMD', 'redis-cli', 'ping'],
                'interval': '30s',
                'timeout': '5s',
                'retries': 3
            }
        }
        
        # PostgreSQL service
        compose_config['services']['postgresql'] = {
            'image': self.services['postgresql']['image'],
            'container_name': f"{self.config['app_name']}-postgres",
            'restart': 'unless-stopped',
            'environment': [
                'POSTGRES_DB=telegram_bot',
                'POSTGRES_USER=telegram_bot',
                'POSTGRES_PASSWORD=${DB_PASSWORD}',
                f"POSTGRES_MAX_CONNECTIONS={self.services['postgresql']['config']['max_connections']}"
            ],
            'ports': ['5432:5432'],
            'volumes': [
                'postgres-data:/var/lib/postgresql/data',
                './database/init.sql:/docker-entrypoint-initdb.d/init.sql:ro'
            ],
            'networks': ['telegram-bot-network'],
            'healthcheck': {
                'test': ['CMD-SHELL', 'pg_isready -U telegram_bot -d telegram_bot'],
                'interval': '30s',
                'timeout': '5s',
                'retries': 3
            }
        }
        
        # Prometheus monitoring
        compose_config['services']['prometheus'] = {
            'image': self.services['prometheus']['image'],
            'container_name': f"{self.config['app_name']}-prometheus",
            'restart': 'unless-stopped',
            'ports': ['9090:9090'],
            'volumes': [
                'prometheus-data:/prometheus',
                './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro'
            ],
            'networks': ['telegram-bot-network'],
            'command': [
                '--config.file=/etc/prometheus/prometheus.yml',
                '--storage.tsdb.path=/prometheus',
                '--web.console.libraries=/etc/prometheus/console_libraries',
                '--web.console.templates=/etc/prometheus/consoles',
                '--storage.tsdb.retention.time=15d',
                '--web.enable-lifecycle'
            ]
        }
        
        # Grafana dashboards
        compose_config['services']['grafana'] = {
            'image': self.services['grafana']['image'],
            'container_name': f"{self.config['app_name']}-grafana",
            'restart': 'unless-stopped',
            'ports': ['3000:3000'],
            'volumes': [
                'grafana-data:/var/lib/grafana',
                './monitoring/dashboards:/etc/grafana/provisioning/dashboards:ro',
                './monitoring/datasources:/etc/grafana/provisioning/datasources:ro'
            ],
            'networks': ['telegram-bot-network'],
            'environment': [
                'GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}',
                'GF_USERS_ALLOW_SIGN_UP=false'
            ]
        }
        
        # Nginx reverse proxy
        compose_config['services']['nginx'] = {
            'image': 'nginx:alpine',
            'container_name': f"{self.config['app_name']}-nginx",
            'restart': 'unless-stopped',
            'ports': ['80:80', '443:443'],
            'volumes': [
                './nginx/nginx.conf:/etc/nginx/nginx.conf:ro',
                './nginx/ssl:/etc/nginx/ssl:ro'
            ],
            'networks': ['telegram-bot-network'],
            'depends_on': ['telegram-bot']
        }
        
        # Write docker-compose.yml
        compose_file = self.project_root / 'docker-compose.prod.yml'
        with open(compose_file, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Generated docker-compose.prod.yml")
    
    def create_monitoring_config(self) -> None:
        """Create monitoring configuration files."""
        monitoring_dir = self.project_root / 'monitoring'
        monitoring_dir.mkdir(exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'scrape_configs': [
                {
                    'job_name': 'telegram-bot',
                    'static_configs': [
                        {'targets': ['telegram-bot:8000']}
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'redis',
                    'static_configs': [
                        {'targets': ['redis:6379']}
                    ]
                },
                {
                    'job_name': 'postgresql',
                    'static_configs': [
                        {'targets': ['postgresql:5432']}
                    ]
                }
            ]
        }
        
        with open(monitoring_dir / 'prometheus.yml', 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        # Grafana datasource configuration
        datasources_dir = monitoring_dir / 'datasources'
        datasources_dir.mkdir(exist_ok=True)
        
        grafana_datasource = {
            'apiVersion': 1,
            'datasources': [
                {
                    'name': 'Prometheus',
                    'type': 'prometheus',
                    'access': 'proxy',
                    'url': 'http://prometheus:9090',
                    'isDefault': True
                }
            ]
        }
        
        with open(datasources_dir / 'prometheus.yml', 'w') as f:
            yaml.dump(grafana_datasource, f, default_flow_style=False)
        
        logger.info("Created monitoring configuration files")
    
    def create_nginx_config(self) -> None:
        """Create Nginx reverse proxy configuration."""
        nginx_dir = self.project_root / 'nginx'
        nginx_dir.mkdir(exist_ok=True)
        
        nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream telegram_bot {
        server telegram-bot:8000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=webhook:10m rate=100r/s;
    
    server {
        listen 80;
        server_name _;
        
        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://telegram_bot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Webhook endpoint
        location /webhook/ {
            limit_req zone=webhook burst=50 nodelay;
            proxy_pass http://telegram_bot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            proxy_pass http://telegram_bot;
            access_log off;
        }
        
        # Metrics (restrict access)
        location /metrics {
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
            
            proxy_pass http://telegram_bot;
        }
    }
}
"""
        
        with open(nginx_dir / 'nginx.conf', 'w') as f:
            f.write(nginx_config.strip())
        
        logger.info("Created Nginx configuration")
    
    def create_systemd_service(self) -> None:
        """Create systemd service file for production deployment."""
        service_config = f"""
[Unit]
Description=Telegram ML Bot
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory={self.project_root}
ExecStart=/usr/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.project_root / f'{self.config["app_name"]}.service'
        with open(service_file, 'w') as f:
            f.write(service_config.strip())
        
        console.print(f"[green]✓ Created systemd service file: {service_file}[/green]")
        console.print(f"[yellow]To install, run:[/yellow]")
        console.print(f"  sudo cp {service_file} /etc/systemd/system/")
        console.print(f"  sudo systemctl daemon-reload")
        console.print(f"  sudo systemctl enable {self.config['app_name']}")
    
    def create_environment_template(self) -> None:
        """Create .env template for production."""
        env_template = """
# Production Environment Configuration

# Application Settings
ENVIRONMENT=production
DEBUG=false
APP_NAME=Telegram ML Bot
APP_VERSION=1.0.0

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database Configuration
DB_HOST=postgresql
DB_PORT=5432
DB_NAME=telegram_bot
DB_USER=telegram_bot
DB_PASSWORD=CHANGE_ME_STRONG_PASSWORD
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=CHANGE_ME_REDIS_PASSWORD
REDIS_MAX_CONNECTIONS=50

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_FROM_BOTFATHER
TELEGRAM_WEBHOOK_URL=https://your-domain.com/webhook/telegram
TELEGRAM_WEBHOOK_SECRET=CHANGE_ME_WEBHOOK_SECRET
TELEGRAM_RATE_LIMIT_CALLS=20
TELEGRAM_RATE_LIMIT_PERIOD=60

# Security Settings
SECRET_KEY=CHANGE_ME_VERY_LONG_RANDOM_STRING
JWT_SECRET_KEY=CHANGE_ME_ANOTHER_LONG_RANDOM_STRING
JWT_ALGORITHM=HS256
JWT_EXPIRATION_SECONDS=3600

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# CORS Settings
CORS_ORIGINS=https://your-domain.com,https://api.your-domain.com

# Monitoring Settings
LOG_LEVEL=INFO
LOG_FORMAT=json
METRICS_ENABLED=true
SENTRY_DSN=YOUR_SENTRY_DSN_IF_USING_SENTRY

# Grafana Settings
GRAFANA_PASSWORD=CHANGE_ME_GRAFANA_PASSWORD

# SSL/TLS (if using custom certificates)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem
"""
        
        env_file = self.project_root / '.env.production'
        with open(env_file, 'w') as f:
            f.write(env_template.strip())
        
        console.print(f"[green]✓ Created environment template: {env_file}[/green]")
        console.print("[yellow]⚠ Remember to fill in all the CHANGE_ME values![/yellow]")
    
    async def deploy(self, environment: str = 'production') -> None:
        """Deploy the Telegram bot to production."""
        console.print(Panel(
            f"[bold green]Deploying Telegram Bot to {environment.upper()}[/bold green]",
            border_style="green"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            # Initialize Docker
            task1 = progress.add_task("Initializing Docker...", total=100)
            await self.initialize_docker()
            progress.update(task1, advance=100)
            
            # Create configuration files
            task2 = progress.add_task("Creating configuration files...", total=100)
            self.create_docker_compose()
            progress.update(task2, advance=25)
            
            self.create_monitoring_config()
            progress.update(task2, advance=25)
            
            self.create_nginx_config()
            progress.update(task2, advance=25)
            
            self.create_systemd_service()
            progress.update(task2, advance=25)
            
            # Create environment template
            task3 = progress.add_task("Creating environment template...", total=100)
            self.create_environment_template()
            progress.update(task3, advance=100)
            
            # Build Docker images
            task4 = progress.add_task("Building Docker images...", total=100)
            try:
                # Build main application image
                subprocess.run([
                    'docker', 'build', '-t', 
                    f"{self.config['app_name']}:{self.config['version']}", 
                    '.'
                ], cwd=self.project_root, check=True)
                progress.update(task4, advance=100)
                
            except subprocess.CalledProcessError as e:
                console.print(f"[red]✗ Docker build failed: {e}[/red]")
                return
        
        console.print("\n[bold green]✓ Deployment preparation completed successfully![/bold green]")
        
        # Display next steps
        next_steps = Tree("📋 Next Steps")
        
        config_step = next_steps.add("1. Configure Environment")
        config_step.add("• Edit .env.production with your actual values")
        config_step.add("• Generate strong passwords for all CHANGE_ME fields")
        config_step.add("• Set up your Telegram bot token from @BotFather")
        config_step.add("• Configure your webhook URL (must be HTTPS)")
        
        deploy_step = next_steps.add("2. Deploy Services")
        deploy_step.add("• docker-compose -f docker-compose.prod.yml up -d")
        deploy_step.add("• Check logs: docker-compose -f docker-compose.prod.yml logs -f")
        deploy_step.add("• Install systemd service (optional)")
        
        verify_step = next_steps.add("3. Verify Deployment")
        verify_step.add("• Health check: curl http://localhost:8000/health")
        verify_step.add("• Metrics: http://localhost:9090 (Prometheus)")
        verify_step.add("• Dashboard: http://localhost:3000 (Grafana)")
        verify_step.add("• Test webhook: curl -X POST http://localhost:8000/webhook/telegram")
        
        monitor_step = next_steps.add("4. Monitor & Maintain")
        monitor_step.add("• Set up log rotation")
        monitor_step.add("• Configure backup strategies")
        monitor_step.add("• Set up alerting rules")
        monitor_step.add("• Regular security updates")
        
        console.print(next_steps)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'overall': True,
            'services': {},
            'system': {}
        }
        
        try:
            # Check Docker services
            if self.docker_client:
                containers = self.docker_client.containers.list()
                for container in containers:
                    if self.config['app_name'] in container.name:
                        health_status['services'][container.name] = {
                            'status': container.status,
                            'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown')
                        }
            
            # Check system resources
            health_status['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Check if any service is unhealthy
            for service_name, service_info in health_status['services'].items():
                if service_info['status'] != 'running' or service_info['health'] == 'unhealthy':
                    health_status['overall'] = False
            
            # Check system resource thresholds
            if (health_status['system']['cpu_percent'] > 80 or 
                health_status['system']['memory_percent'] > 80 or
                health_status['system']['disk_percent'] > 90):
                health_status['overall'] = False
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            health_status['overall'] = False
            health_status['error'] = str(e)
        
        return health_status
    
    async def scale_services(self, replicas: int) -> None:
        """Scale the bot services."""
        try:
            subprocess.run([
                'docker-compose', '-f', 'docker-compose.prod.yml', 
                'scale', f'telegram-bot={replicas}'
            ], cwd=self.project_root, check=True)
            
            console.print(f"[green]✓ Scaled telegram-bot to {replicas} replicas[/green]")
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗ Scaling failed: {e}[/red]")


@click.group()
@click.pass_context
def cli(ctx):
    """Production deployment CLI for Telegram Bot."""
    ctx.ensure_object(dict)
    ctx.obj['deployer'] = TelegramBotDeployer()


@cli.command()
@click.option('--environment', default='production', help='Deployment environment')
@click.pass_context
async def deploy(ctx, environment: str):
    """Deploy the Telegram bot to production."""
    deployer = ctx.obj['deployer']
    await deployer.deploy(environment)


@cli.command()
@click.pass_context
async def health(ctx):
    """Check health status of deployed services."""
    deployer = ctx.obj['deployer']
    await deployer.initialize_docker()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Checking health...", total=None)
        health_status = await deployer.health_check()
        progress.update(task, description="Health check complete ✓")
    
    # Display health status
    if health_status['overall']:
        console.print("[green]✓ All services are healthy[/green]")
    else:
        console.print("[red]✗ Some services are unhealthy[/red]")
    
    # Services table
    if health_status.get('services'):
        services_table = Table(title="Service Status", show_header=True)
        services_table.add_column("Service", style="cyan")
        services_table.add_column("Status", style="green")
        services_table.add_column("Health", style="blue")
        
        for service_name, service_info in health_status['services'].items():
            status_color = "green" if service_info['status'] == 'running' else "red"
            health_color = "green" if service_info['health'] == 'healthy' else "red"
            
            services_table.add_row(
                service_name,
                f"[{status_color}]{service_info['status']}[/{status_color}]",
                f"[{health_color}]{service_info['health']}[/{health_color}]"
            )
        
        console.print(services_table)
    
    # System resources
    system_info = health_status.get('system', {})
    if system_info:
        system_table = Table(title="System Resources", show_header=True)
        system_table.add_column("Resource", style="cyan")
        system_table.add_column("Usage", style="green")
        system_table.add_column("Status", style="blue")
        
        for resource, value in system_info.items():
            if resource in ['cpu_percent', 'memory_percent', 'disk_percent']:
                if value > 80:
                    status = "[red]High[/red]"
                elif value > 60:
                    status = "[yellow]Medium[/yellow]"
                else:
                    status = "[green]Normal[/green]"
                
                system_table.add_row(
                    resource.replace('_', ' ').title(),
                    f"{value:.1f}%",
                    status
                )
        
        console.print(system_table)


@cli.command()
@click.option('--replicas', default=3, help='Number of replicas to scale to')
@click.pass_context
async def scale(ctx, replicas: int):
    """Scale the bot services."""
    deployer = ctx.obj['deployer']
    await deployer.scale_services(replicas)


def run_async_command(coro):
    """Run async command in event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.create_task(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# Make async commands work with click
for command in [deploy, health, scale]:
    command.callback = lambda ctx, *args, **kwargs, cmd=command.callback: run_async_command(cmd(ctx, *args, **kwargs))


if __name__ == '__main__':
    cli()