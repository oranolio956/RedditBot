#!/usr/bin/env python3
"""
Secrets Migration Script

Migrates sensitive configuration from .env file to encrypted secrets storage.
This script helps transition from hardcoded secrets to secure secrets management.

Usage:
    python scripts/migrate_secrets.py [--dry-run] [--force]
    
Options:
    --dry-run: Show what would be migrated without making changes
    --force: Overwrite existing secrets in encrypted storage
    --backup: Create backup of current .env file
"""

import argparse
import asyncio
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.core.secrets_manager import get_secrets_manager, ProductionSecretsManager
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Define which environment variables contain sensitive data
SENSITIVE_KEYS = {
    'TELEGRAM_BOT_TOKEN',
    'TELEGRAM_WEBHOOK_SECRET', 
    'SECRET_KEY',
    'JWT_SECRET_KEY',
    'DATABASE_PASSWORD',
    'DB_PASSWORD',
    'REDIS_PASSWORD',
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'STRIPE_SECRET_KEY',
    'STRIPE_WEBHOOK_SECRET',
    'SENTRY_DSN',
    'ENCRYPTION_KEY',
    'SECRETS_MASTER_PASSWORD'
}

# Non-sensitive keys that should remain in .env
NON_SENSITIVE_KEYS = {
    'ENVIRONMENT',
    'DEBUG',
    'TESTING',
    'APP_NAME',
    'APP_VERSION',
    'API_PREFIX',
    'HOST',
    'PORT',
    'WORKERS',
    'DB_HOST',
    'DB_PORT',
    'DB_NAME',
    'DB_USER',
    'REDIS_HOST',
    'REDIS_PORT',
    'REDIS_DB',
    'JWT_ALGORITHM',
    'JWT_EXPIRATION_SECONDS',
    'RATE_LIMIT_ENABLED',
    'RATE_LIMIT_PER_MINUTE',
    'LOG_LEVEL',
    'LOG_FORMAT',
    'METRICS_ENABLED'
}


class SecretsMigrator:
    """Handles migration of secrets from .env to encrypted storage."""
    
    def __init__(self, env_file: str = ".env", dry_run: bool = False, force: bool = False):
        self.env_file = Path(env_file)
        self.dry_run = dry_run
        self.force = force
        self.secrets_manager: ProductionSecretsManager = None
        self.migration_report = {
            'migrated_secrets': [],
            'skipped_secrets': [],
            'errors': [],
            'warnings': []
        }
    
    async def initialize(self):
        """Initialize the secrets manager."""
        try:
            self.secrets_manager = get_secrets_manager()
            logger.info("Secrets manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize secrets manager: {e}")
            raise
    
    def parse_env_file(self) -> Dict[str, str]:
        """Parse the .env file and return key-value pairs."""
        env_vars = {}
        
        if not self.env_file.exists():
            logger.warning(f"Environment file {self.env_file} not found")
            return env_vars
        
        try:
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' not in line:
                        logger.warning(f"Invalid line {line_num} in {self.env_file}: {line}")
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    env_vars[key] = value
            
            logger.info(f"Parsed {len(env_vars)} variables from {self.env_file}")
            return env_vars
            
        except Exception as e:
            logger.error(f"Failed to parse {self.env_file}: {e}")
            raise
    
    def identify_sensitive_vars(self, env_vars: Dict[str, str]) -> Dict[str, Set[str]]:
        """Categorize environment variables into sensitive and non-sensitive."""
        sensitive_vars = {}
        non_sensitive_vars = {}
        unknown_vars = {}
        
        for key, value in env_vars.items():
            if key.upper() in SENSITIVE_KEYS:
                sensitive_vars[key] = value
            elif key.upper() in NON_SENSITIVE_KEYS:
                non_sensitive_vars[key] = value
            else:
                # Use heuristics to detect sensitive data
                if self._is_likely_sensitive(key, value):
                    sensitive_vars[key] = value
                    self.migration_report['warnings'].append(
                        f"Variable {key} detected as sensitive using heuristics"
                    )
                else:
                    unknown_vars[key] = value
                    self.migration_report['warnings'].append(
                        f"Variable {key} categorization unknown - treating as non-sensitive"
                    )
        
        logger.info(
            f"Categorized variables: {len(sensitive_vars)} sensitive, "
            f"{len(non_sensitive_vars)} non-sensitive, {len(unknown_vars)} unknown"
        )
        
        return {
            'sensitive': sensitive_vars,
            'non_sensitive': {**non_sensitive_vars, **unknown_vars},
            'unknown': unknown_vars
        }
    
    def _is_likely_sensitive(self, key: str, value: str) -> bool:
        """Use heuristics to detect if a variable is likely sensitive."""
        key_upper = key.upper()
        
        # Check for sensitive keywords in key name
        sensitive_keywords = {
            'TOKEN', 'SECRET', 'KEY', 'PASSWORD', 'PASS', 'AUTH', 
            'CREDENTIAL', 'API_KEY', 'PRIVATE', 'DSN', 'WEBHOOK'
        }
        
        if any(keyword in key_upper for keyword in sensitive_keywords):
            return True
        
        # Check for sensitive patterns in value
        if not value:
            return False
        
        # Long random-looking strings are likely secrets
        if len(value) > 32 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value):
            return True
        
        # JWT tokens
        if value.startswith(('eyJ', 'ey0')):
            return True
        
        # API key patterns
        if any(value.startswith(prefix) for prefix in ['sk_', 'pk_', 'rk_', 'bot']):
            return True
        
        return False
    
    async def migrate_secrets(self, sensitive_vars: Dict[str, str]) -> bool:
        """Migrate sensitive variables to encrypted storage."""
        success_count = 0
        
        for key, value in sensitive_vars.items():
            try:
                # Check if secret already exists
                existing_value = await self.secrets_manager.get_secret(key.lower())
                
                if existing_value and not self.force:
                    logger.info(f"Secret {key} already exists in encrypted storage (use --force to overwrite)")
                    self.migration_report['skipped_secrets'].append(key)
                    continue
                
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would migrate secret: {key}")
                    self.migration_report['migrated_secrets'].append(key)
                    success_count += 1
                    continue
                
                # Migrate the secret
                success = await self.secrets_manager.set_secret(
                    key.lower(),
                    value,
                    metadata={
                        'migrated_from': 'env_file',
                        'migration_date': datetime.utcnow().isoformat(),
                        'original_key': key
                    }
                )
                
                if success:
                    logger.info(f"Successfully migrated secret: {key}")
                    self.migration_report['migrated_secrets'].append(key)
                    success_count += 1
                else:
                    error_msg = f"Failed to migrate secret: {key}"
                    logger.error(error_msg)
                    self.migration_report['errors'].append(error_msg)
                
            except Exception as e:
                error_msg = f"Error migrating {key}: {e}"
                logger.error(error_msg)
                self.migration_report['errors'].append(error_msg)
        
        logger.info(f"Migration completed: {success_count}/{len(sensitive_vars)} secrets migrated")
        return success_count == len(sensitive_vars)
    
    def create_new_env_file(self, non_sensitive_vars: Dict[str, str], sensitive_vars: Dict[str, str]):
        """Create a new .env file without sensitive data."""
        backup_file = self.env_file.with_suffix('.env.backup')
        new_env_file = self.env_file.with_suffix('.env.new')
        
        try:
            # Create backup
            if self.env_file.exists():
                shutil.copy2(self.env_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would create new .env file: {new_env_file}")
                return
            
            # Create new .env file
            with open(new_env_file, 'w', encoding='utf-8') as f:
                f.write("# Environment Configuration\n")
                f.write("# Sensitive values have been moved to encrypted secrets storage\n")
                f.write(f"# Migration completed: {datetime.utcnow().isoformat()}\n\n")
                
                # Write non-sensitive variables
                for key, value in sorted(non_sensitive_vars.items()):
                    f.write(f"{key}={value}\n")
                
                f.write("\n# Sensitive variables (now in encrypted storage):\n")
                for key in sorted(sensitive_vars.keys()):
                    f.write(f"# {key}=<moved to encrypted storage>\n")
                
                f.write("\n# To access encrypted secrets, use the secrets manager API\n")
                f.write("# or set USE_ENCRYPTED_SECRETS=true\n")
            
            logger.info(f"Created new environment file: {new_env_file}")
            logger.warning("Please review the new .env file and replace the original when ready")
            
        except Exception as e:
            logger.error(f"Failed to create new .env file: {e}")
            raise
    
    def print_migration_report(self):
        """Print a summary of the migration."""
        print("\n" + "="*60)
        print("SECRETS MIGRATION REPORT")
        print("="*60)
        
        if self.migration_report['migrated_secrets']:
            print(f"\n✅ Successfully migrated ({len(self.migration_report['migrated_secrets'])}):") 
            for secret in self.migration_report['migrated_secrets']:
                print(f"   - {secret}")
        
        if self.migration_report['skipped_secrets']:
            print(f"\n⏭️  Skipped ({len(self.migration_report['skipped_secrets'])}):") 
            for secret in self.migration_report['skipped_secrets']:
                print(f"   - {secret} (already exists)")
        
        if self.migration_report['warnings']:
            print(f"\n⚠️  Warnings ({len(self.migration_report['warnings'])}):") 
            for warning in self.migration_report['warnings']:
                print(f"   - {warning}")
        
        if self.migration_report['errors']:
            print(f"\n❌ Errors ({len(self.migration_report['errors'])}):") 
            for error in self.migration_report['errors']:
                print(f"   - {error}")
        
        print("\n" + "="*60)
        
        if self.dry_run:
            print("This was a DRY RUN - no changes were made.")
            print("Run without --dry-run to perform the actual migration.")
        else:
            print("Migration completed!")
            print("Next steps:")
            print("1. Test your application with encrypted secrets")
            print("2. Set USE_ENCRYPTED_SECRETS=true in your environment")
            print("3. Review and replace your .env file")
            print("4. Remove sensitive values from version control")
        
        print("="*60)
    
    async def run_migration(self) -> bool:
        """Run the complete migration process."""
        try:
            logger.info("Starting secrets migration...")
            
            # Parse existing .env file
            env_vars = self.parse_env_file()
            if not env_vars:
                logger.warning("No environment variables found to migrate")
                return True
            
            # Categorize variables
            categorized = self.identify_sensitive_vars(env_vars)
            sensitive_vars = categorized['sensitive']
            non_sensitive_vars = categorized['non_sensitive']
            
            if not sensitive_vars:
                logger.info("No sensitive variables found to migrate")
                return True
            
            logger.info(f"Found {len(sensitive_vars)} sensitive variables to migrate")
            
            # Migrate sensitive variables
            await self.initialize()
            migration_success = await self.migrate_secrets(sensitive_vars)
            
            # Create new .env file
            if migration_success or self.dry_run:
                self.create_new_env_file(non_sensitive_vars, sensitive_vars)
            
            # Print report
            self.print_migration_report()
            
            return migration_success
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False


async def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate secrets from .env file to encrypted storage"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Overwrite existing secrets in encrypted storage"
    )
    parser.add_argument(
        "--env-file", 
        default=".env", 
        help="Path to .env file (default: .env)"
    )
    parser.add_argument(
        "--backup", 
        action="store_true", 
        help="Create backup of .env file before migration"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    if not Path(args.env_file).exists():
        logger.error(f"Environment file {args.env_file} not found")
        return False
    
    # Create migrator and run
    migrator = SecretsMigrator(
        env_file=args.env_file,
        dry_run=args.dry_run,
        force=args.force
    )
    
    success = await migrator.run_migration()
    
    if success:
        logger.info("Secrets migration completed successfully")
        return True
    else:
        logger.error("Secrets migration failed")
        return False


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    exit(0 if success else 1)
