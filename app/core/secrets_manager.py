"""
Production-Ready Secrets Management System

Secure handling of sensitive credentials with multiple provider support,
automated rotation, audit logging, and enterprise security features.
"""

import os
import json
import base64
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod

import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = structlog.get_logger(__name__)


@dataclass
class SecretMetadata:
    """Secret metadata for audit and management."""
    name: str
    created_at: datetime
    updated_at: datetime
    version: str
    rotation_interval: Optional[int] = None  # days
    last_rotated: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class SecretValue:
    """Encrypted secret value with metadata."""
    encrypted_value: str
    metadata: SecretMetadata
    checksum: str
    
    def verify_integrity(self, decrypted_value: str) -> bool:
        """Verify secret integrity using checksum."""
        calculated_checksum = hashlib.sha256(decrypted_value.encode()).hexdigest()
        return self.checksum == calculated_checksum


class SecretProvider(ABC):
    """Abstract base class for secret providers."""
    
    @abstractmethod
    async def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret value."""
        pass
    
    @abstractmethod
    async def set_secret(self, name: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store a secret value."""
        pass
    
    @abstractmethod
    async def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        pass
    
    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List all secret names."""
        pass


class EnvironmentSecretProvider(SecretProvider):
    """Environment variables secret provider with prefix support."""
    
    def __init__(self, prefix: str = "SECRET_"):
        self.prefix = prefix
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from environment variable."""
        env_name = f"{self.prefix}{name.upper()}"
        return os.getenv(env_name)
    
    async def set_secret(self, name: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Set environment variable (runtime only)."""
        env_name = f"{self.prefix}{name.upper()}"
        os.environ[env_name] = value
        return True
    
    async def delete_secret(self, name: str) -> bool:
        """Delete environment variable."""
        env_name = f"{self.prefix}{name.upper()}"
        if env_name in os.environ:
            del os.environ[env_name]
            return True
        return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets with prefix."""
        secrets = []
        for key in os.environ:
            if key.startswith(self.prefix):
                secret_name = key[len(self.prefix):].lower()
                secrets.append(secret_name)
        return secrets


class EncryptedFileSecretProvider(SecretProvider):
    """Encrypted file-based secret provider for secure local storage."""
    
    def __init__(self, secrets_file: str = ".secrets.enc", master_password: Optional[str] = None):
        self.secrets_file = Path(secrets_file)
        self.master_password = master_password or os.getenv("SECRETS_MASTER_PASSWORD")
        self.cipher_suite = None
        self.secrets_cache: Dict[str, SecretValue] = {}
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption with master password."""
        try:
            if not self.master_password:
                # Generate a new master password and warn user
                self.master_password = secrets.token_urlsafe(32)
                logger.warning(
                    "No master password provided. Generated new one. "
                    f"Save this securely: {self.master_password}"
                )
            
            # Derive key from password
            password_bytes = self.master_password.encode()
            salt = b'salt_for_secrets'  # In production, use random salt stored separately
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            self.cipher_suite = Fernet(key)
            
            # Load existing secrets
            self._load_secrets()
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _load_secrets(self):
        """Load and decrypt secrets from file."""
        if not self.secrets_file.exists():
            return
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            if not encrypted_data:
                return
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            secrets_data = json.loads(decrypted_data.decode())
            
            # Reconstruct SecretValue objects
            for name, data in secrets_data.items():
                metadata = SecretMetadata(**data['metadata'])
                # Convert datetime strings back to datetime objects
                metadata.created_at = datetime.fromisoformat(metadata.created_at)
                metadata.updated_at = datetime.fromisoformat(metadata.updated_at)
                if metadata.last_rotated:
                    metadata.last_rotated = datetime.fromisoformat(metadata.last_rotated)
                if metadata.last_accessed:
                    metadata.last_accessed = datetime.fromisoformat(metadata.last_accessed)
                
                self.secrets_cache[name] = SecretValue(
                    encrypted_value=data['encrypted_value'],
                    metadata=metadata,
                    checksum=data['checksum']
                )
            
            logger.info(f"Loaded {len(self.secrets_cache)} secrets from encrypted storage")
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
    
    def _save_secrets(self):
        """Encrypt and save secrets to file."""
        try:
            # Convert to serializable format
            secrets_data = {}
            for name, secret_value in self.secrets_cache.items():
                metadata_dict = asdict(secret_value.metadata)
                # Convert datetime objects to strings
                metadata_dict['created_at'] = secret_value.metadata.created_at.isoformat()
                metadata_dict['updated_at'] = secret_value.metadata.updated_at.isoformat()
                if secret_value.metadata.last_rotated:
                    metadata_dict['last_rotated'] = secret_value.metadata.last_rotated.isoformat()
                if secret_value.metadata.last_accessed:
                    metadata_dict['last_accessed'] = secret_value.metadata.last_accessed.isoformat()
                
                secrets_data[name] = {
                    'encrypted_value': secret_value.encrypted_value,
                    'metadata': metadata_dict,
                    'checksum': secret_value.checksum
                }
            
            # Encrypt and save
            data_bytes = json.dumps(secrets_data).encode()
            encrypted_data = self.cipher_suite.encrypt(data_bytes)
            
            # Ensure directory exists
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with restricted permissions
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set file permissions to read/write for owner only
            os.chmod(self.secrets_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get and decrypt secret value."""
        try:
            if name not in self.secrets_cache:
                return None
            
            secret_value = self.secrets_cache[name]
            
            # Decrypt value
            decrypted_bytes = self.cipher_suite.decrypt(secret_value.encrypted_value.encode())
            decrypted_value = decrypted_bytes.decode()
            
            # Verify integrity
            if not secret_value.verify_integrity(decrypted_value):
                logger.error(f"Secret integrity check failed for: {name}")
                return None
            
            # Update access metadata
            secret_value.metadata.access_count += 1
            secret_value.metadata.last_accessed = datetime.utcnow()
            self._save_secrets()
            
            return decrypted_value
            
        except Exception as e:
            logger.error(f"Failed to get secret {name}: {e}")
            return None
    
    async def set_secret(self, name: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Encrypt and store secret value."""
        try:
            # Encrypt value
            encrypted_bytes = self.cipher_suite.encrypt(value.encode())
            encrypted_value = encrypted_bytes.decode()
            
            # Create checksum
            checksum = hashlib.sha256(value.encode()).hexdigest()
            
            # Create or update metadata
            now = datetime.utcnow()
            if name in self.secrets_cache:
                # Update existing
                existing_metadata = self.secrets_cache[name].metadata
                existing_metadata.updated_at = now
                existing_metadata.version = str(int(existing_metadata.version) + 1)
                if metadata:
                    existing_metadata.tags.update(metadata)
                secret_metadata = existing_metadata
            else:
                # Create new
                secret_metadata = SecretMetadata(
                    name=name,
                    created_at=now,
                    updated_at=now,
                    version="1",
                    tags=metadata or {}
                )
            
            # Store secret
            self.secrets_cache[name] = SecretValue(
                encrypted_value=encrypted_value,
                metadata=secret_metadata,
                checksum=checksum
            )
            
            # Save to file
            self._save_secrets()
            
            logger.info(f"Secret stored: {name} (version {secret_metadata.version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret {name}: {e}")
            return False
    
    async def delete_secret(self, name: str) -> bool:
        """Delete secret from storage."""
        try:
            if name in self.secrets_cache:
                del self.secrets_cache[name]
                self._save_secrets()
                logger.info(f"Secret deleted: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete secret {name}: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        """List all secret names."""
        return list(self.secrets_cache.keys())


class ProductionSecretsManager:
    """Production-ready secrets manager with multiple providers and security features."""
    
    def __init__(self):
        self.providers: List[SecretProvider] = []
        self.audit_log: List[Dict[str, Any]] = []
        self.secret_cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(minutes=5)
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize secret providers in order of preference."""
        # 1. Encrypted file storage (primary for development)
        if os.getenv("USE_ENCRYPTED_SECRETS", "true").lower() == "true":
            self.providers.append(EncryptedFileSecretProvider())
        
        # 2. Environment variables (fallback)
        self.providers.append(EnvironmentSecretProvider())
        
        logger.info(f"Initialized {len(self.providers)} secret providers")
    
    def _log_access(self, action: str, secret_name: str, success: bool, details: Optional[str] = None):
        """Log secret access for audit purposes."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "secret_name": secret_name,
            "success": success,
            "details": details
        }
        self.audit_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        # Log to structured logger
        logger.info(
            f"Secret {action}",
            secret_name=secret_name,
            success=success,
            details=details
        )
    
    async def get_secret(self, name: str, use_cache: bool = True) -> Optional[str]:
        """Get secret value from first available provider."""
        try:
            # Check cache first
            if use_cache and name in self.secret_cache:
                cache_entry = self.secret_cache[name]
                if datetime.utcnow() - cache_entry["timestamp"] < self.cache_ttl:
                    self._log_access("get", name, True, "from_cache")
                    return cache_entry["value"]
            
            # Try providers in order
            for provider in self.providers:
                try:
                    value = await provider.get_secret(name)
                    if value is not None:
                        # Cache the value
                        if use_cache:
                            self.secret_cache[name] = {
                                "value": value,
                                "timestamp": datetime.utcnow()
                            }
                        
                        self._log_access("get", name, True, f"from_{provider.__class__.__name__}")
                        return value
                except Exception as e:
                    logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
            
            self._log_access("get", name, False, "not_found")
            return None
            
        except Exception as e:
            self._log_access("get", name, False, str(e))
            logger.error(f"Failed to get secret {name}: {e}")
            return None
    
    async def set_secret(
        self, 
        name: str, 
        value: str, 
        provider_index: int = 0,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Set secret value in specified provider."""
        try:
            if provider_index >= len(self.providers):
                raise ValueError(f"Invalid provider index: {provider_index}")
            
            provider = self.providers[provider_index]
            success = await provider.set_secret(name, value, metadata)
            
            if success:
                # Invalidate cache
                self.secret_cache.pop(name, None)
                self._log_access("set", name, True, f"in_{provider.__class__.__name__}")
            else:
                self._log_access("set", name, False, "provider_failed")
            
            return success
            
        except Exception as e:
            self._log_access("set", name, False, str(e))
            logger.error(f"Failed to set secret {name}: {e}")
            return False
    
    async def delete_secret(self, name: str, all_providers: bool = True) -> bool:
        """Delete secret from providers."""
        try:
            success = False
            
            if all_providers:
                # Delete from all providers
                for provider in self.providers:
                    try:
                        if await provider.delete_secret(name):
                            success = True
                    except Exception as e:
                        logger.warning(f"Failed to delete from {provider.__class__.__name__}: {e}")
            else:
                # Delete from first provider only
                success = await self.providers[0].delete_secret(name)
            
            if success:
                # Remove from cache
                self.secret_cache.pop(name, None)
                self._log_access("delete", name, True, "deleted")
            else:
                self._log_access("delete", name, False, "not_found")
            
            return success
            
        except Exception as e:
            self._log_access("delete", name, False, str(e))
            logger.error(f"Failed to delete secret {name}: {e}")
            return False
    
    async def list_all_secrets(self) -> Dict[str, List[str]]:
        """List secrets from all providers."""
        all_secrets = {}
        
        for i, provider in enumerate(self.providers):
            try:
                secrets = await provider.list_secrets()
                all_secrets[f"provider_{i}_{provider.__class__.__name__}"] = secrets
            except Exception as e:
                logger.error(f"Failed to list secrets from {provider.__class__.__name__}: {e}")
        
        return all_secrets
    
    async def validate_required_secrets(self) -> Dict[str, bool]:
        """Validate that all required secrets are present."""
        required_secrets = [
            "TELEGRAM_BOT_TOKEN",
            "JWT_SECRET",
            "SECRET_KEY",
            "DATABASE_PASSWORD",
            "REDIS_PASSWORD"
        ]
        
        validation_results = {}
        
        for secret_name in required_secrets:
            value = await self.get_secret(secret_name)
            validation_results[secret_name] = value is not None and len(value) > 0
            
            if not validation_results[secret_name]:
                logger.warning(f"Required secret missing or empty: {secret_name}")
        
        return validation_results
    
    async def rotate_secret(self, name: str, new_value: Optional[str] = None) -> bool:
        """Rotate a secret with a new value."""
        try:
            if new_value is None:
                # Generate a new secure value
                new_value = secrets.token_urlsafe(32)
            
            # Store new value
            success = await self.set_secret(name, new_value, metadata={"rotated": "true"})
            
            if success:
                self._log_access("rotate", name, True, "rotated")
            else:
                self._log_access("rotate", name, False, "rotation_failed")
            
            return success
            
        except Exception as e:
            self._log_access("rotate", name, False, str(e))
            logger.error(f"Failed to rotate secret {name}: {e}")
            return False
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        return self.audit_log[-limit:]
    
    def clear_cache(self):
        """Clear the secret cache."""
        self.secret_cache.clear()
        logger.info("Secret cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        valid_entries = 0
        expired_entries = 0
        
        for entry in self.secret_cache.values():
            if now - entry["timestamp"] < self.cache_ttl:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(self.secret_cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_ttl_minutes": self.cache_ttl.total_seconds() / 60
        }


# Global secrets manager instance
_secrets_manager: Optional[ProductionSecretsManager] = None


def get_secrets_manager() -> ProductionSecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = ProductionSecretsManager()
    return _secrets_manager


async def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    manager = get_secrets_manager()
    value = await manager.get_secret(name)
    return value if value is not None else default


async def set_secret(name: str, value: str, metadata: Optional[Dict] = None) -> bool:
    """Convenience function to set a secret."""
    manager = get_secrets_manager()
    return await manager.set_secret(name, value, metadata=metadata)


async def validate_all_secrets() -> bool:
    """Validate all required secrets are present."""
    manager = get_secrets_manager()
    validation = await manager.validate_required_secrets()
    return all(validation.values())


# Migration helper for moving from .env to encrypted storage
async def migrate_env_to_encrypted():
    """Migrate secrets from .env file to encrypted storage."""
    try:
        env_file = Path(".env")
        if not env_file.exists():
            logger.info("No .env file found to migrate")
            return
        
        manager = get_secrets_manager()
        migrated_count = 0
        
        with open(env_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                
                # Only migrate sensitive keys
                sensitive_keys = [
                    'TELEGRAM_BOT_TOKEN', 'JWT_SECRET', 'SECRET_KEY',
                    'DATABASE_PASSWORD', 'REDIS_PASSWORD', 'STRIPE_SECRET_KEY',
                    'STRIPE_WEBHOOK_SECRET', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'
                ]
                
                if any(sensitive in key.upper() for sensitive in sensitive_keys):
                    if await manager.set_secret(key.lower(), value, {"migrated_from": "env"}):
                        migrated_count += 1
                        logger.info(f"Migrated secret: {key}")
        
        logger.info(f"Migration complete. Migrated {migrated_count} secrets")
        logger.warning(
            "Please update your application to use the encrypted secrets "
            "and remove sensitive values from .env file"
        )
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


# Export main components
__all__ = [
    "ProductionSecretsManager",
    "EncryptedFileSecretProvider",
    "EnvironmentSecretProvider",
    "get_secrets_manager",
    "get_secret",
    "set_secret",
    "validate_all_secrets",
    "migrate_env_to_encrypted"
]
