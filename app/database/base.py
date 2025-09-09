"""
Database Base Models and Mixins

Provides base classes and mixins for SQLAlchemy models with
common fields, methods, and utilities for the application.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, String, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr, declarative_base
from sqlalchemy.sql import func
import structlog

logger = structlog.get_logger(__name__)


class TimestampMixin:
    """Mixin for adding timestamp fields to models."""
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Record creation timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
        comment="Record last update timestamp"
    )


class UUIDMixin:
    """Mixin for adding UUID primary key to models."""
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False,
        comment="Unique identifier"
    )


class SoftDeleteMixin:
    """Mixin for adding soft delete functionality to models."""
    
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Soft delete timestamp"
    )
    
    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Soft delete flag"
    )
    
    def soft_delete(self):
        """Mark record as deleted without removing from database."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self):
        """Restore soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """Mixin for adding audit fields to track who created/modified records."""
    
    created_by = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="ID of user who created this record"
    )
    
    updated_by = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="ID of user who last updated this record"
    )


class Base:
    """Base class with common functionality for all models."""
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    def to_dict(self, exclude: Optional[list] = None) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.
        
        Args:
            exclude: List of field names to exclude from output
            
        Returns:
            Dictionary representation of the model
        """
        exclude = exclude or []
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                
                # Handle special types
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                
                result[column.name] = value
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[list] = None):
        """
        Update model instance from dictionary.
        
        Args:
            data: Dictionary with field names and values
            exclude: List of field names to exclude from update
        """
        exclude = exclude or ['id', 'created_at']
        
        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        """String representation of model instance."""
        if hasattr(self, 'id'):
            return f"<{self.__class__.__name__}(id={self.id})>"
        return f"<{self.__class__.__name__}>"


# Create declarative base with our custom Base class
DeclarativeBase = declarative_base(cls=Base)


class BaseModel(DeclarativeBase, UUIDMixin, TimestampMixin):
    """
    Base model class with UUID primary key and timestamps.
    
    This is the recommended base class for most models in the application.
    It provides:
    - UUID primary key
    - created_at and updated_at timestamps
    - Common utility methods
    """
    
    __abstract__ = True
    
    class Config:
        """Pydantic configuration for serialization."""
        from_attributes = True
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v),
        }


class AuditableModel(BaseModel, AuditMixin):
    """
    Base model with audit trail functionality.
    
    Extends BaseModel with fields to track who created
    and last modified each record.
    """
    
    __abstract__ = True


class SoftDeletableModel(BaseModel, SoftDeleteMixin):
    """
    Base model with soft delete functionality.
    
    Extends BaseModel with soft delete capabilities,
    allowing records to be marked as deleted without
    physical removal from the database.
    """
    
    __abstract__ = True


class FullAuditModel(BaseModel, AuditMixin, SoftDeleteMixin):
    """
    Base model with full audit and soft delete functionality.
    
    Combines all audit features including:
    - UUID primary key
    - Timestamps
    - Audit trail (who created/modified)
    - Soft delete capability
    """
    
    __abstract__ = True