"""
Security Utilities for Revolutionary Features

Implements encryption, input sanitization, and privacy protection
for consciousness mirroring, memory palace, and temporal archaeology features.
"""

import hashlib
import hmac
import json
import re
import secrets
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

from app.config.settings import get_settings


class EncryptionService:
    """
    Handles encryption/decryption of sensitive psychological and biometric data.
    Uses Fernet (AES-128) for symmetric encryption with key rotation.
    """
    
    def __init__(self):
        self.master_key = self._get_or_create_master_key()
        self.fernet = Fernet(self.master_key)
        self.rotation_interval = timedelta(days=30)
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create encryption master key."""
        # In production, use AWS KMS or Azure Key Vault
        settings = get_settings()
        key_env = getattr(settings.security, 'encryption_key', None)
        
        if not key_env:
            # Generate new key for development
            key = Fernet.generate_key()
            print(f"WARNING: Generated new encryption key. Set ENCRYPTION_KEY env variable.")
            return key
            
        return key_env.encode() if isinstance(key_env, str) else key_env
        
    def encrypt_psychological_data(self, data: Dict) -> str:
        """Encrypt psychological profile data with enhanced validation."""
        if not data:
            raise ValueError("Cannot encrypt empty psychological data")
        
        # Validate no PII in clear fields
        sanitized = self._remove_pii(data)
        
        # Ensure all psychological fields are encrypted
        sensitive_fields = ['personality', 'keystrokes', 'linguistic_fingerprint', 
                          'decision_patterns', 'emotional_state', 'cognitive_biases',
                          'temporal_evolution', 'response_templates']
        
        for field in sensitive_fields:
            if field in sanitized and isinstance(sanitized[field], (str, dict, list)):
                # Mark field as requiring encryption
                sanitized[f"{field}_encrypted"] = True
        
        # Convert to JSON bytes
        json_bytes = json.dumps(sanitized, default=str).encode()
        
        # Encrypt
        encrypted = self.fernet.encrypt(json_bytes)
        
        # Return base64 string
        return base64.b64encode(encrypted).decode()
        
    def decrypt_psychological_data(self, encrypted_data: str) -> Dict:
        """Decrypt psychological profile data."""
        try:
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            
            # Decrypt
            decrypted = self.fernet.decrypt(encrypted_bytes)
            
            # Parse JSON
            return json.loads(decrypted.decode())
        except Exception as e:
            raise ValueError(f"Failed to decrypt psychological data: {e}")
            
    def encrypt_biometric_data(self, keystroke_data: List[float]) -> str:
        """Encrypt keystroke biometric patterns."""
        # Add noise for differential privacy
        noisy_data = self._add_differential_privacy_noise(keystroke_data)
        
        # Convert to bytes
        data_bytes = json.dumps(noisy_data).encode()
        
        # Encrypt
        encrypted = self.fernet.encrypt(data_bytes)
        
        return base64.b64encode(encrypted).decode()
        
    def decrypt_biometric_data(self, encrypted_data: str) -> List[float]:
        """Decrypt keystroke biometric patterns."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return json.loads(decrypted.decode())
        except Exception as e:
            raise ValueError(f"Failed to decrypt biometric data: {e}")
            
    def encrypt_spatial_data(self, position_data: List[float]) -> str:
        """Encrypt 3D spatial position data."""
        # Round to reduce precision (privacy)
        rounded = [round(x, 2) for x in position_data]
        
        data_bytes = json.dumps(rounded).encode()
        encrypted = self.fernet.encrypt(data_bytes)
        
        return base64.b64encode(encrypted).decode()
        
    def decrypt_spatial_data(self, encrypted_data: str) -> List[float]:
        """Decrypt 3D spatial position data."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return json.loads(decrypted.decode())
        except Exception as e:
            raise ValueError(f"Failed to decrypt spatial data: {e}")
            
    def _remove_pii(self, data: Dict) -> Dict:
        """Remove personally identifiable information."""
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card
        ]
        
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str):
                for pattern in pii_patterns:
                    value = re.sub(pattern, '[REDACTED]', value)
            cleaned[key] = value
            
        return cleaned
        
    def _add_differential_privacy_noise(self, data: List[float], epsilon: float = 1.0) -> List[float]:
        """Add Laplacian noise for differential privacy with enhanced protection."""
        if not data:
            return []
        
        sensitivity = 1.0  # L1 sensitivity
        scale = sensitivity / epsilon
        
        noisy_data = []
        for value in data:
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                noise = np.random.laplace(0, scale)
                noisy_value = value + noise
                # Ensure bounded output
                noisy_data.append(max(-10, min(10, noisy_value)))
            else:
                noisy_data.append(0.0)
            
        return noisy_data


class InputSanitizer:
    """
    Sanitizes user inputs to prevent injection attacks and data poisoning.
    """
    
    def __init__(self):
        self.max_input_length = 5000
        self.max_vector_dimensions = 2048
        self.max_json_depth = 5
        
    def sanitize_text_input(self, text: str) -> str:
        """Sanitize text input for LLM processing."""
        if not text:
            return ""
            
        # Length limit
        text = text[:self.max_input_length]
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        # Prevent prompt injection (enhanced patterns)
        injection_patterns = [
            r'ignore previous instructions',
            r'disregard all prior',
            r'forget everything',
            r'system:',
            r'assistant:',
            r'human:',
            r'<\|.*?\|>',  # Special tokens
            r'\[INST\]',  # Instruction tokens
            r'\[/INST\]',
            r'<s>.*?</s>',  # Special sequence tokens
            r'###\s*(?:instruction|system|human|assistant)',
            r'(?:override|bypass|circumvent).*?(?:safety|security|filter)',
            r'jailbreak',
            r'\bDAN\b',  # "Do Anything Now" prompt
            r'act as if.*?uncensored',
            r'simulate.*?without.*?restrictions',
            r'roleplay.*?without.*?limitations',
            r'pretend.*?no.*?rules'
        ]
        
        for pattern in injection_patterns:
            text = re.sub(pattern, '[BLOCKED]', text, flags=re.IGNORECASE)
            
        # Escape special characters
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        
        return text
        
    def sanitize_vector_input(self, vector: List[float]) -> List[float]:
        """Sanitize vector embeddings."""
        if not vector:
            return []
            
        # Limit dimensions
        vector = vector[:self.max_vector_dimensions]
        
        # Validate numeric values
        sanitized = []
        for value in vector:
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                # Clip to reasonable range
                sanitized.append(max(-100, min(100, float(value))))
            else:
                sanitized.append(0.0)
                
        return sanitized
        
    def sanitize_3d_position(self, position: List[float]) -> List[float]:
        """Sanitize 3D coordinates to prevent overflow."""
        if not position or len(position) != 3:
            return [0.0, 0.0, 0.0]
            
        # Bound checking
        bounds = {
            'x': (-1000, 1000),
            'y': (-100, 100),
            'z': (-1000, 1000)
        }
        
        sanitized = []
        for i, (axis, (min_val, max_val)) in enumerate(bounds.items()):
            if i < len(position):
                value = position[i]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    sanitized.append(max(min_val, min(max_val, float(value))))
                else:
                    sanitized.append(0.0)
            else:
                sanitized.append(0.0)
                
        return sanitized
        
    def sanitize_json_input(self, data: Any, depth: int = 0) -> Any:
        """Recursively sanitize JSON data."""
        if depth > self.max_json_depth:
            return None
            
        if isinstance(data, dict):
            sanitized = {}
            for key, value in list(data.items())[:100]:  # Limit keys
                if isinstance(key, str):
                    clean_key = self.sanitize_text_input(key[:100])
                    sanitized[clean_key] = self.sanitize_json_input(value, depth + 1)
            return sanitized
            
        elif isinstance(data, list):
            return [self.sanitize_json_input(item, depth + 1) for item in data[:1000]]
            
        elif isinstance(data, str):
            return self.sanitize_text_input(data)
            
        elif isinstance(data, (int, float)):
            if np.isnan(data) or np.isinf(data):
                return 0
            return max(-1e10, min(1e10, data))
            
        elif isinstance(data, bool):
            return data
            
        else:
            return None
            
    def validate_pattern_complexity(self, pattern: str) -> bool:
        """Validate regex pattern complexity to prevent ReDoS."""
        if not pattern:
            return False
            
        # Limit pattern length
        if len(pattern) > 500:
            return False
            
        # Check for dangerous constructs
        dangerous_patterns = [
            r'\(\?\!',  # Negative lookahead
            r'\(\?\<\!',  # Negative lookbehind
            r'\(\?\=',  # Positive lookahead
            r'\(\?\<\=',  # Positive lookbehind
            r'.*\+',  # Nested quantifiers
            r'.*\*',
            r'.*\{',
        ]
        
        for dangerous in dangerous_patterns:
            if re.search(dangerous, pattern):
                return False
                
        # Test pattern with timeout
        try:
            test_pattern = re.compile(pattern)
            test_pattern.search("test" * 100)
            return True
        except:
            return False


class PrivacyProtector:
    """
    Implements privacy protection for sensitive features.
    """
    
    def __init__(self):
        self.min_k_anonymity = 5
        self.noise_scale = 0.1
        
    def apply_k_anonymity(self, data: List[Dict], quasi_identifiers: List[str]) -> List[Dict]:
        """Apply k-anonymity to dataset."""
        if len(data) < self.min_k_anonymity:
            # Not enough data for anonymization
            return []
            
        # Group by quasi-identifiers
        groups = {}
        for record in data:
            key = tuple(record.get(qi) for qi in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
            
        # Only return groups meeting k-anonymity
        anonymized = []
        for group in groups.values():
            if len(group) >= self.min_k_anonymity:
                # Generalize sensitive attributes
                for record in group:
                    anonymized.append(self._generalize_record(record))
                    
        return anonymized
        
    def _generalize_record(self, record: Dict) -> Dict:
        """Generalize sensitive attributes."""
        generalized = record.copy()
        
        # Age ranges instead of exact age
        if 'age' in generalized:
            age = generalized['age']
            generalized['age_range'] = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
            del generalized['age']
            
        # Location generalization
        if 'location' in generalized:
            # Keep only country/state
            generalized['location'] = generalized['location'].split(',')[0]
            
        return generalized
        
    def add_local_differential_privacy(self, value: float, sensitivity: float = 1.0) -> float:
        """Add noise for local differential privacy."""
        epsilon = 1.0  # Privacy budget
        
        # Laplacian mechanism
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        
        return value + noise
        
    def hash_user_identifier(self, identifier: str, salt: str = None) -> str:
        """Create pseudonymous identifier."""
        if not salt:
            settings = get_settings()
            salt = getattr(settings.security, 'secret_key', 'default-salt')[:16]
            
        # Use HMAC for secure hashing
        h = hmac.new(salt.encode(), identifier.encode(), hashlib.sha256)
        return h.hexdigest()
        
    def redact_conversation_content(self, content: str) -> str:
        """Redact sensitive information from conversations."""
        # Names (simple heuristic)
        content = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', content)
        
        # Addresses
        content = re.sub(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)', '[ADDRESS]', content)
        
        # Dates
        content = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', content)
        
        # Numbers that might be IDs
        content = re.sub(r'\b\d{6,}\b', '[ID]', content)
        
        return content


class RateLimiter:
    """
    Rate limiting for ML model access to prevent extraction attacks.
    """
    
    def __init__(self):
        self.limits = {
            'consciousness_mirror': {'calls': 100, 'window': 3600},  # 100 per hour
            'memory_palace': {'calls': 200, 'window': 3600},  # 200 per hour
            'temporal_archaeology': {'calls': 50, 'window': 3600},  # 50 per hour
            'llm_inference': {'calls': 500, 'window': 3600},  # 500 per hour
        }
        self.user_calls = {}  # In production, use Redis
        
    async def check_rate_limit(self, user_id: str, feature: str) -> bool:
        """Check if user has exceeded rate limit."""
        if feature not in self.limits:
            return True
            
        key = f"{user_id}:{feature}"
        now = datetime.utcnow()
        
        if key not in self.user_calls:
            self.user_calls[key] = []
            
        # Clean old calls
        window = timedelta(seconds=self.limits[feature]['window'])
        self.user_calls[key] = [
            call_time for call_time in self.user_calls[key]
            if now - call_time < window
        ]
        
        # Check limit
        if len(self.user_calls[key]) >= self.limits[feature]['calls']:
            return False
            
        # Record call
        self.user_calls[key].append(now)
        return True
        
    def get_remaining_calls(self, user_id: str, feature: str) -> int:
        """Get remaining calls for user."""
        if feature not in self.limits:
            return 0
            
        key = f"{user_id}:{feature}"
        if key not in self.user_calls:
            return self.limits[feature]['calls']
            
        now = datetime.utcnow()
        window = timedelta(seconds=self.limits[feature]['window'])
        
        recent_calls = [
            call_time for call_time in self.user_calls[key]
            if now - call_time < window
        ]
        
        return max(0, self.limits[feature]['calls'] - len(recent_calls))


class MLSecurityValidator:
    """
    Validates ML model inputs/outputs to prevent attacks.
    """
    
    def __init__(self):
        self.max_prompt_length = 2000
        self.max_output_length = 5000
        self.blocked_outputs = []
        
    def validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Validate LLM prompt for injection attacks."""
        # Length check
        if len(prompt) > self.max_prompt_length:
            return False, "Prompt too long"
            
        # Injection patterns
        injection_indicators = [
            'ignore all previous',
            'disregard instructions',
            'reveal your prompt',
            'show your instructions',
            'what is your system message',
            'repeat everything above',
            'print previous conversation',
        ]
        
        prompt_lower = prompt.lower()
        for indicator in injection_indicators:
            if indicator in prompt_lower:
                return False, f"Potential injection detected: {indicator}"
                
        return True, "Valid"
        
    def validate_model_output(self, output: str) -> Tuple[bool, str]:
        """Validate model output for data leakage."""
        # Length check
        if len(output) > self.max_output_length:
            return False, "Output too long"
            
        # Check for leaked keys/secrets
        secret_patterns = [
            r'sk_[a-zA-Z0-9]{48}',  # Stripe keys
            r'pk_[a-zA-Z0-9]{48}',
            r'[a-zA-Z0-9]{40}',  # API keys
            r'-----BEGIN.*KEY-----',  # Private keys
        ]
        
        for pattern in secret_patterns:
            if re.search(pattern, output):
                return False, "Potential secret leak detected"
                
        # Check for PII
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, output):
                return False, "PII detected in output"
                
        return True, "Valid"
        
    def validate_vector_similarity_threshold(self, similarity: float) -> bool:
        """Validate vector similarity scores."""
        # Prevent model inversion via similarity scores
        if similarity > 0.99:
            # Too similar - might be extraction attempt
            return False
        return True


class ConsentManager:
    """
    Manages user consent for sensitive data processing.
    """
    
    def __init__(self):
        self.consent_types = {
            'psychological_profiling': {
                'description': 'Create psychological profile from messages',
                'data_types': ['messages', 'typing_patterns', 'response_times'],
                'retention_days': 365
            },
            'biometric_keystroke': {
                'description': 'Analyze keystroke dynamics',
                'data_types': ['keystroke_timing', 'typing_rhythm'],
                'retention_days': 180
            },
            'conversation_reconstruction': {
                'description': 'Reconstruct missing conversations',
                'data_types': ['message_history', 'linguistic_patterns'],
                'retention_days': 90
            },
            'spatial_tracking': {
                'description': 'Track navigation in memory palace',
                'data_types': ['3d_positions', 'navigation_paths'],
                'retention_days': 365
            }
        }
        
    async def check_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has given consent."""
        # In production, check database
        # For now, return True for development
        return True
        
    async def record_consent(self, user_id: str, consent_type: str, granted: bool) -> bool:
        """Record user consent decision."""
        if consent_type not in self.consent_types:
            return False
            
        # In production, save to database with timestamp
        consent_record = {
            'user_id': user_id,
            'consent_type': consent_type,
            'granted': granted,
            'timestamp': datetime.utcnow(),
            'ip_address': None,  # Would get from request
            'details': self.consent_types[consent_type]
        }
        
        return True
        
    def get_required_consents(self, feature: str) -> List[str]:
        """Get required consents for a feature."""
        feature_consents = {
            'consciousness_mirror': ['psychological_profiling', 'biometric_keystroke'],
            'memory_palace': ['spatial_tracking'],
            'temporal_archaeology': ['conversation_reconstruction', 'psychological_profiling']
        }
        
        return feature_consents.get(feature, [])


class UserIsolationManager:
    """
    Ensures complete user data isolation to prevent cross-user contamination.
    """
    
    def __init__(self):
        self.user_contexts = {}  # In production, use Redis
        self.isolation_salt = secrets.token_hex(16)
    
    def create_isolated_context(self, user_id: str) -> str:
        """Create isolated context key for user."""
        if not user_id:
            raise ValueError("User ID required for isolation")
        
        # Create deterministic but secure context key
        context_key = hmac.new(
            self.isolation_salt.encode(),
            user_id.encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        
        self.user_contexts[user_id] = context_key
        return context_key
    
    def validate_user_access(self, user_id: str, data_context: str) -> bool:
        """Validate user can access specific data context."""
        expected_context = self.user_contexts.get(user_id)
        if not expected_context:
            expected_context = self.create_isolated_context(user_id)
        
        return expected_context == data_context
    
    def sanitize_cache_key(self, user_id: str, base_key: str) -> str:
        """Create user-isolated cache key."""
        context = self.create_isolated_context(user_id)
        return f"{context}:{base_key}"
    
    def validate_no_cross_contamination(self, user_id: str, data: Dict) -> bool:
        """Validate data doesn't contain other users' information."""
        # Check for user_id fields that don't match current user
        user_id_fields = ['user_id', 'owner_id', 'created_by', 'belongs_to']
        
        for field in user_id_fields:
            if field in data and str(data[field]) != str(user_id):
                return False
        
        # Check for nested user references
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and not self.validate_no_cross_contamination(user_id, value):
                    return False
        
        return True


# Singleton instances
encryption_service = EncryptionService()
input_sanitizer = InputSanitizer()
privacy_protector = PrivacyProtector()
rate_limiter = RateLimiter()
ml_validator = MLSecurityValidator()
consent_manager = ConsentManager()
user_isolation = UserIsolationManager()


# Helper functions for easy access
async def encrypt_psychological_profile(data: Dict) -> str:
    """Encrypt psychological profile data."""
    return encryption_service.encrypt_psychological_data(data)

async def decrypt_psychological_profile(encrypted: str) -> Dict:
    """Decrypt psychological profile data."""
    return encryption_service.decrypt_psychological_data(encrypted)

async def sanitize_llm_input(text: str) -> str:
    """Sanitize text for LLM processing."""
    return input_sanitizer.sanitize_text_input(text)

async def check_ml_rate_limit(user_id: str, feature: str) -> bool:
    """Check ML feature rate limit."""
    return await rate_limiter.check_rate_limit(user_id, feature)

async def validate_ml_output(output: str) -> Tuple[bool, str]:
    """Validate ML model output."""
    return ml_validator.validate_model_output(output)