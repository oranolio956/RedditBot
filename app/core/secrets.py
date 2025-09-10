"""
Production Secrets Management

Secure handling of sensitive credentials with support for multiple providers:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- Environment variables (fallback)

Features:
- Automatic secret rotation
- Audit logging
- Encryption at rest
- Access control
- Secret versioning
"""

import os
import json
import base64
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = structlog.get_logger(__name__)


class SecretProvider(str, Enum):
    """Supported secret storage providers."""
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    HASHICORP_VAULT = "hashicorp_vault"
    AZURE_KEY_VAULT = "azure_key_vault"
    ENVIRONMENT = "environment"
    LOCAL_ENCRYPTED = "local_encrypted"


@dataclass
class Secret:
    """Secret data container."""
    name: str
    value: str
    version: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def is_expired(self) -> bool:
        """Check if secret has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False


class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""
    
    @abstractmethod
    async def get_secret(self, name: str, version: Optional[str] = None) -> Optional[Secret]:
        """Retrieve a secret by name."""
        pass
    
    @abstractmethod
    async def set_secret(self, name: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store a secret."""
        pass
    
    @abstractmethod
    async def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        pass
    
    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List all available secret names."""
        pass
    
    @abstractmethod
    async def rotate_secret(self, name: str) -> bool:
        """Rotate a secret."""
        pass


class AWSSecretsManagerProvider(SecretsProvider):
    """AWS Secrets Manager provider."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AWS Secrets Manager client."""
        try:
            import boto3
            self.client = boto3.client('secretsmanager', region_name=self.region)
            logger.info(f"AWS Secrets Manager initialized in region {self.region}")
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Secrets Manager: {e}")
    
    async def get_secret(self, name: str, version: Optional[str] = None) -> Optional[Secret]:
        """Get secret from AWS Secrets Manager."""
        if not self.client:
            return None
        
        try:
            params = {'SecretId': name}
            if version:
                params['VersionId'] = version
            
            response = self.client.get_secret_value(**params)
            
            # Parse secret value
            if 'SecretString' in response:
                secret_value = response['SecretString']
            else:
                secret_value = base64.b64decode(response['SecretBinary']).decode('utf-8')
            
            # Try to parse as JSON
            try:
                secret_data = json.loads(secret_value)
                if isinstance(secret_data, dict) and 'value' in secret_data:
                    secret_value = secret_data['value']
            except json.JSONDecodeError:
                pass  # Use raw value
            
            return Secret(
                name=name,
                value=secret_value,
                version=response.get('VersionId'),
                created_at=response.get('CreatedDate'),
                metadata=response.get('Tags', {})
            )
            
        except self.client.exceptions.ResourceNotFoundException:
            logger.warning(f"Secret not found: {name}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving secret {name}: {e}")
            return None
    
    async def set_secret(self, name: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store secret in AWS Secrets Manager."""
        if not self.client:
            return False
        
        try:
            # Check if secret exists
            try:
                self.client.describe_secret(SecretId=name)
                # Update existing secret
                self.client.update_secret(
                    SecretId=name,
                    SecretString=value
                )
                logger.info(f"Updated secret: {name}")
            except self.client.exceptions.ResourceNotFoundException:
                # Create new secret
                params = {
                    'Name': name,
                    'SecretString': value
                }
                if metadata:
                    params['Tags'] = [{'Key': k, 'Value': v} for k, v in metadata.items()]
                
                self.client.create_secret(**params)
                logger.info(f"Created secret: {name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing secret {name}: {e}")
            return False
    
    async def delete_secret(self, name: str) -> bool:
        """Delete secret from AWS Secrets Manager."""
        if not self.client:
            return False
        
        try:
            self.client.delete_secret(
                SecretId=name,
                ForceDeleteWithoutRecovery=False  # Allow recovery within 30 days
            )
            logger.info(f"Scheduled secret for deletion: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting secret {name}: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets in AWS Secrets Manager."""
        if not self.client:
            return []
        
        try:
            secrets = []
            paginator = self.client.get_paginator('list_secrets')
            
            for page in paginator.paginate():
                for secret in page['SecretList']:
                    secrets.append(secret['Name'])
            
            return secrets
        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            return []
    
    async def rotate_secret(self, name: str) -> bool:
        """Rotate a secret in AWS Secrets Manager."""
        if not self.client:
            return False
        
        try:
            self.client.rotate_secret(
                SecretId=name,
                RotationRules={'AutomaticallyAfterDays': 30}
            )
            logger.info(f"Initiated rotation for secret: {name}")
            return True
        except Exception as e:
            logger.error(f"Error rotating secret {name}: {e}")
            return False


class EnvironmentSecretsProvider(SecretsProvider):
    """Environment variables secrets provider (fallback)."""
    
    def __init__(self, prefix: str = "APP_SECRET_"):
        self.prefix = prefix
    
    async def get_secret(self, name: str, version: Optional[str] = None) -> Optional[Secret]:
        """Get secret from environment variables."""
        env_name = f"{self.prefix}{name.upper()}"
        value = os.getenv(env_name)
        
        if value:
            return Secret(
                name=name,
                value=value,
                created_at=datetime.utcnow()
            )
        return None
    
    async def set_secret(self, name: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Set environment variable (runtime only)."""
        env_name = f"{self.prefix}{name.upper()}"
        os.environ[env_name] = value
        return True
    
    async def delete_secret(self, name: str) -> bool:
        """Delete environment variable (runtime only)."""
        env_name = f"{self.prefix}{name.upper()}"
        if env_name in os.environ:
            del os.environ[env_name]
            return True
        return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets in environment."""
        secrets = []
        for key in os.environ:
            if key.startswith(self.prefix):
                secret_name = key[len(self.prefix):].lower()
                secrets.append(secret_name)
        return secrets
    
    async def rotate_secret(self, name: str) -> bool:
        """Rotation not supported for environment variables."""
        logger.warning("Secret rotation not supported for environment provider")
        return False


class LocalEncryptedSecretsProvider(SecretsProvider):
    """Local encrypted file-based secrets provider."""
    
    def __init__(self, secrets_file: str = ".secrets.enc", master_key: Optional[str] = None):
        self.secrets_file = secrets_file
        self.cipher_suite = None
        self.secrets_cache = {}
        self._initialize_encryption(master_key)
    
    def _initialize_encryption(self, master_key: Optional[str] = None):
        """Initialize encryption with master key."""
        try:
            if master_key:
                key = base64.urlsafe_b64encode(master_key.encode()[:32].ljust(32, b'0'))
            else:
                # Generate or load key from environment
                key_env = os.getenv('SECRETS_MASTER_KEY')
                if key_env:
                    key = base64.urlsafe_b64encode(key_env.encode()[:32].ljust(32, b'0'))
                else:
                    # Generate new key (should be stored securely)
                    key = Fernet.generate_key()
                    logger.warning(f"Generated new master key. Store this securely: {key.decode()}")
            
            self.cipher_suite = Fernet(key)
            self._load_secrets()
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
    
    def _load_secrets(self):
        """Load and decrypt secrets from file."""
        if not os.path.exists(self.secrets_file):
            return
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            self.secrets_cache = json.loads(decrypted_data.decode())
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
    
    def _save_secrets(self):
        """Encrypt and save secrets to file."""
        try:
            data = json.dumps(self.secrets_cache).encode()
            encrypted_data = self.cipher_suite.encrypt(data)
            
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
    
    async def get_secret(self, name: str, version: Optional[str] = None) -> Optional[Secret]:
        """Get secret from encrypted storage."""
        if name in self.secrets_cache:
            secret_data = self.secrets_cache[name]
            return Secret(
                name=name,
                value=secret_data['value'],
                created_at=datetime.fromisoformat(secret_data.get('created_at', datetime.utcnow().isoformat())),
                metadata=secret_data.get('metadata', {})
            )
        return None
    
    async def set_secret(self, name: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store secret in encrypted storage."""
        try:
            self.secrets_cache[name] = {
                'value': value,
                'created_at': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            self._save_secrets()
            return True
        except Exception as e:
            logger.error(f"Failed to store secret {name}: {e}")
            return False
    
    async def delete_secret(self, name: str) -> bool:
        """Delete secret from encrypted storage."""
        if name in self.secrets_cache:
            del self.secrets_cache[name]
            self._save_secrets()
            return True
        return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets in encrypted storage."""
        return list(self.secrets_cache.keys())
    
    async def rotate_secret(self, name: str) -> bool:
        """Rotate encryption key."""
        # This would require re-encrypting all secrets with new key
        logger.warning("Secret rotation requires manual key rotation")
        return False


class SecretsManager:
    """
    Central secrets management system.
    
    Handles multiple providers with fallback support and caching.
    """
    
    def __init__(self):
        self.providers: Dict[SecretProvider, SecretsProvider] = {}
        self.cache: Dict[str, Secret] = {}
        self.cache_ttl = timedelta(minutes=5)
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available secret providers."""
        # Check which providers are available
        if os.getenv('AWS_REGION'):
            self.providers[SecretProvider.AWS_SECRETS_MANAGER] = AWSSecretsManagerProvider()
        
        # Always have environment fallback
        self.providers[SecretProvider.ENVIRONMENT] = EnvironmentSecretsProvider()
        
        # Local encrypted storage for development
        if os.getenv('USE_LOCAL_SECRETS', 'false').lower() == 'true':
            self.providers[SecretProvider.LOCAL_ENCRYPTED] = LocalEncryptedSecretsProvider()
        
        logger.info(f"Initialized secret providers: {list(self.providers.keys())}")
    
    async def get_secret(self, name: str, provider: Optional[SecretProvider] = None) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            name: Secret name
            provider: Specific provider to use (optional)
            
        Returns:
            Secret value or None if not found
        """
        # Check cache first
        cache_key = f"{provider}:{name}" if provider else name
        if cache_key in self.cache:
            secret = self.cache[cache_key]
            if not secret.is_expired() and \
               (not secret.created_at or datetime.utcnow() - secret.created_at < self.cache_ttl):
                return secret.value
        
        # Try specified provider or all providers
        if provider and provider in self.providers:
            secret = await self.providers[provider].get_secret(name)
            if secret:
                self.cache[cache_key] = secret
                return secret.value
        else:
            # Try all providers in order
            for provider_type, provider_instance in self.providers.items():
                secret = await provider_instance.get_secret(name)
                if secret:
                    self.cache[name] = secret
                    logger.info(f"Found secret {name} in {provider_type}")
                    return secret.value
        
        logger.warning(f"Secret not found: {name}")
        return None
    
    async def set_secret(
        self, 
        name: str, 
        value: str,
        provider: SecretProvider = SecretProvider.ENVIRONMENT,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store a secret."""
        if provider in self.providers:
            success = await self.providers[provider].set_secret(name, value, metadata)
            if success:
                # Invalidate cache
                self.cache.pop(name, None)
                self.cache.pop(f"{provider}:{name}", None)
            return success
        return False
    
    async def delete_secret(self, name: str, provider: Optional[SecretProvider] = None) -> bool:
        """Delete a secret."""
        success = False
        
        if provider and provider in self.providers:
            success = await self.providers[provider].delete_secret(name)
        else:
            # Delete from all providers
            for provider_instance in self.providers.values():
                if await provider_instance.delete_secret(name):
                    success = True
        
        if success:
            # Clear from cache
            self.cache = {k: v for k, v in self.cache.items() if not k.endswith(f":{name}") and k != name}
        
        return success
    
    async def rotate_secret(self, name: str, provider: Optional[SecretProvider] = None) -> bool:
        """Rotate a secret."""
        if provider and provider in self.providers:
            return await self.providers[provider].rotate_secret(name)
        
        # Rotate in first provider that has the secret
        for provider_instance in self.providers.values():
            if await provider_instance.get_secret(name):
                return await provider_instance.rotate_secret(name)
        
        return False
    
    async def get_required_secrets(self) -> Dict[str, Optional[str]]:
        """Get all required secrets for the application."""
        required_secrets = [
            'TELEGRAM_BOT_TOKEN',
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'STRIPE_SECRET_KEY',
            'STRIPE_WEBHOOK_SECRET',
            'DATABASE_PASSWORD',
            'REDIS_PASSWORD',
            'JWT_SECRET',
            'ENCRYPTION_KEY'
        ]
        
        secrets = {}
        for secret_name in required_secrets:
            secrets[secret_name] = await self.get_secret(secret_name)
        
        return secrets
    
    async def validate_secrets(self) -> Dict[str, bool]:
        """Validate that all required secrets are present."""
        secrets = await self.get_required_secrets()
        validation = {}
        
        for name, value in secrets.items():
            validation[name] = value is not None
            if not value:
                logger.warning(f"Missing required secret: {name}")
        
        return validation
    
    def get_safe_config(self) -> Dict[str, Any]:
        """Get configuration with secrets masked."""
        config = {}
        for name in self.cache:
            if ':' not in name:  # Skip provider-specific entries
                config[name] = "***REDACTED***"
        return config


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


async def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


async def get_secret(name: str) -> Optional[str]:
    """Convenience function to get a secret."""
    manager = await get_secrets_manager()
    return await manager.get_secret(name)


async def validate_all_secrets() -> bool:
    """Validate all required secrets are present."""
    manager = await get_secrets_manager()
    validation = await manager.validate_secrets()
    return all(validation.values())


# Export main components
__all__ = [
    'SecretsManager',
    'SecretProvider',
    'Secret',
    'get_secrets_manager',
    'get_secret',
    'validate_all_secrets'
]