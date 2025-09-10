"""
Database Repository Layer

Implements the Repository pattern for data access with advanced querying,
caching, and transaction management capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, Generic, TypeVar, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.sql import Select
import structlog

from app.database.connection import db_manager, get_db_session
from app.database.base import BaseModel
from app.core.redis import redis_manager

logger = structlog.get_logger(__name__)

# Type variables for generic repository
T = TypeVar('T', bound=BaseModel)
CreateSchemaType = TypeVar('CreateSchemaType')
UpdateSchemaType = TypeVar('UpdateSchemaType')


class FilterOperator:
    """Filter operators for dynamic querying."""
    EQ = "eq"           # Equal
    NE = "ne"           # Not equal
    GT = "gt"           # Greater than
    GE = "ge"           # Greater or equal
    LT = "lt"           # Less than
    LE = "le"           # Less or equal
    LIKE = "like"       # String contains
    ILIKE = "ilike"     # Case-insensitive contains
    IN = "in"           # In list
    NOT_IN = "not_in"   # Not in list
    IS_NULL = "is_null" # Is null
    IS_NOT_NULL = "is_not_null" # Is not null


class QueryFilter:
    """Query filter specification."""
    
    def __init__(self, field: str, operator: str, value: Any = None):
        self.field = field
        self.operator = operator
        self.value = value
    
    def apply(self, query: Select, model_class: Type[T]) -> Select:
        """Apply filter to SQLAlchemy query."""
        field_attr = getattr(model_class, self.field)
        
        if self.operator == FilterOperator.EQ:
            return query.where(field_attr == self.value)
        elif self.operator == FilterOperator.NE:
            return query.where(field_attr != self.value)
        elif self.operator == FilterOperator.GT:
            return query.where(field_attr > self.value)
        elif self.operator == FilterOperator.GE:
            return query.where(field_attr >= self.value)
        elif self.operator == FilterOperator.LT:
            return query.where(field_attr < self.value)
        elif self.operator == FilterOperator.LE:
            return query.where(field_attr <= self.value)
        elif self.operator == FilterOperator.LIKE:
            return query.where(field_attr.like(f"%{self.value}%"))
        elif self.operator == FilterOperator.ILIKE:
            return query.where(field_attr.ilike(f"%{self.value}%"))
        elif self.operator == FilterOperator.IN:
            return query.where(field_attr.in_(self.value))
        elif self.operator == FilterOperator.NOT_IN:
            return query.where(~field_attr.in_(self.value))
        elif self.operator == FilterOperator.IS_NULL:
            return query.where(field_attr.is_(None))
        elif self.operator == FilterOperator.IS_NOT_NULL:
            return query.where(field_attr.is_not(None))
        else:
            raise ValueError(f"Unsupported filter operator: {self.operator}")


class SortOrder:
    """Sort order specification."""
    
    def __init__(self, field: str, direction: str = "asc"):
        self.field = field
        self.direction = direction.lower()
    
    def apply(self, query: Select, model_class: Type[T]) -> Select:
        """Apply sort order to SQLAlchemy query."""
        field_attr = getattr(model_class, self.field)
        
        if self.direction == "desc":
            return query.order_by(desc(field_attr))
        else:
            return query.order_by(asc(field_attr))


class PaginationParams:
    """Pagination parameters."""
    
    def __init__(self, page: int = 1, size: int = 50, max_size: int = 1000):
        self.page = max(1, page)
        self.size = min(max(1, size), max_size)
        self.offset = (self.page - 1) * self.size


class QueryResult(Generic[T]):
    """Query result with pagination metadata."""
    
    def __init__(self, items: List[T], total: int, pagination: PaginationParams):
        self.items = items
        self.total = total
        self.page = pagination.page
        self.size = pagination.size
        self.pages = (total + pagination.size - 1) // pagination.size
        self.has_next = pagination.page < self.pages
        self.has_prev = pagination.page > 1


class BaseRepository(Generic[T], ABC):
    """
    Base repository class with common CRUD operations and advanced querying.
    
    Implements caching, performance monitoring, and transaction management.
    """
    
    def __init__(self, model_class: Type[T]):
        self.model_class = model_class
        self.model_name = model_class.__name__.lower()
        
    # Cache management
    
    def _get_cache_key(self, operation: str, *args) -> str:
        """Generate cache key for operation."""
        key_parts = [self.model_name, operation] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            return await redis_manager.get_json(key)
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None
    
    async def _set_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache."""
        try:
            await redis_manager.set_json(key, value, expire=ttl)
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
    
    async def _invalidate_cache_pattern(self, pattern: str) -> None:
        """Invalidate cache keys matching pattern."""
        try:
            await redis_manager.delete_pattern(pattern)
        except Exception as e:
            logger.warning("Cache invalidation failed", pattern=pattern, error=str(e))
    
    # Core CRUD operations
    
    async def get_by_id(self, id: Any, use_cache: bool = True) -> Optional[T]:
        """Get entity by ID with optional caching."""
        cache_key = self._get_cache_key("by_id", id)
        
        if use_cache:
            cached = await self._get_from_cache(cache_key)
            if cached:
                return self.model_class(**cached)
        
        async with db_manager.get_async_session() as session:
            result = await session.execute(
                select(self.model_class).where(self.model_class.id == id)
            )
            entity = result.scalar_one_or_none()
            
            if entity and use_cache:
                await self._set_cache(cache_key, entity.to_dict())
            
            return entity
    
    async def get_by_field(self, field: str, value: Any, use_cache: bool = True) -> Optional[T]:
        """Get entity by specific field value."""
        cache_key = self._get_cache_key("by_field", field, value)
        
        if use_cache:
            cached = await self._get_from_cache(cache_key)
            if cached:
                return self.model_class(**cached)
        
        async with db_manager.get_async_session() as session:
            field_attr = getattr(self.model_class, field)
            result = await session.execute(
                select(self.model_class).where(field_attr == value)
            )
            entity = result.scalar_one_or_none()
            
            if entity and use_cache:
                await self._set_cache(cache_key, entity.to_dict())
            
            return entity
    
    async def create(self, data: Union[Dict[str, Any], CreateSchemaType]) -> T:
        """Create new entity."""
        if not isinstance(data, dict):
            data = data.dict() if hasattr(data, 'dict') else data.__dict__
        
        async with db_manager.get_async_session() as session:
            entity = self.model_class(**data)
            session.add(entity)
            await session.commit()
            await session.refresh(entity)
            
            # Invalidate related caches
            await self._invalidate_cache_pattern(f"{self.model_name}:*")
            
            return entity
    
    async def update(self, id: Any, data: Union[Dict[str, Any], UpdateSchemaType]) -> Optional[T]:
        """Update entity by ID."""
        if not isinstance(data, dict):
            data = data.dict(exclude_unset=True) if hasattr(data, 'dict') else data.__dict__
        
        async with db_manager.get_async_session() as session:
            # Remove None values and empty strings
            clean_data = {k: v for k, v in data.items() if v is not None and v != ""}
            
            if not clean_data:
                return await self.get_by_id(id, use_cache=False)
            
            # Update timestamp
            clean_data['updated_at'] = datetime.utcnow()
            
            result = await session.execute(
                update(self.model_class)
                .where(self.model_class.id == id)
                .values(**clean_data)
                .returning(self.model_class)
            )
            
            entity = result.scalar_one_or_none()
            if entity:
                await session.commit()
                
                # Invalidate caches
                await self._invalidate_cache_pattern(f"{self.model_name}:*")
            
            return entity
    
    async def delete(self, id: Any, soft_delete: bool = True) -> bool:
        """Delete entity (soft delete by default)."""
        async with db_manager.get_async_session() as session:
            if soft_delete and hasattr(self.model_class, 'soft_delete'):
                # Soft delete
                result = await session.execute(
                    update(self.model_class)
                    .where(self.model_class.id == id)
                    .values(is_deleted=True, deleted_at=datetime.utcnow())
                    .returning(self.model_class.id)
                )
            else:
                # Hard delete
                result = await session.execute(
                    delete(self.model_class)
                    .where(self.model_class.id == id)
                    .returning(self.model_class.id)
                )
            
            deleted_id = result.scalar_one_or_none()
            if deleted_id:
                await session.commit()
                
                # Invalidate caches
                await self._invalidate_cache_pattern(f"{self.model_name}:*")
                
                return True
            
            return False
    
    # Advanced querying
    
    async def find_by_filters(
        self,
        filters: List[QueryFilter] = None,
        sort_orders: List[SortOrder] = None,
        pagination: PaginationParams = None,
        include_deleted: bool = False,
        use_cache: bool = False
    ) -> QueryResult[T]:
        """Find entities with complex filtering, sorting, and pagination."""
        
        filters = filters or []
        sort_orders = sort_orders or [SortOrder("created_at", "desc")]
        pagination = pagination or PaginationParams()
        
        # Build cache key for query
        cache_key = None
        if use_cache:
            cache_parts = [
                f"filters:{len(filters)}",
                f"sorts:{len(sort_orders)}",
                f"page:{pagination.page}",
                f"size:{pagination.size}",
                f"deleted:{include_deleted}"
            ]
            cache_key = self._get_cache_key("find", *cache_parts)
            
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                items = [self.model_class(**item) for item in cached_result['items']]
                return QueryResult(items, cached_result['total'], pagination)
        
        async with db_manager.get_async_session() as session:
            # Base query
            query = select(self.model_class)
            count_query = select(func.count(self.model_class.id))
            
            # Apply soft delete filter
            if not include_deleted and hasattr(self.model_class, 'is_deleted'):
                query = query.where(self.model_class.is_deleted == False)
                count_query = count_query.where(self.model_class.is_deleted == False)
            
            # Apply filters
            for filter_spec in filters:
                query = filter_spec.apply(query, self.model_class)
                count_query = filter_spec.apply(count_query, self.model_class)
            
            # Get total count
            total_result = await session.execute(count_query)
            total = total_result.scalar()
            
            # Apply sorting
            for sort_order in sort_orders:
                query = sort_order.apply(query, self.model_class)
            
            # Apply pagination
            query = query.offset(pagination.offset).limit(pagination.size)
            
            # Execute query
            result = await session.execute(query)
            items = result.scalars().all()
            
            # Cache results
            if use_cache and cache_key:
                cache_data = {
                    'items': [item.to_dict() for item in items],
                    'total': total
                }
                await self._set_cache(cache_key, cache_data, ttl=600)  # 10 minutes
            
            return QueryResult(items, total, pagination)
    
    async def count_by_filters(self, filters: List[QueryFilter] = None, include_deleted: bool = False) -> int:
        """Count entities matching filters."""
        filters = filters or []
        
        async with db_manager.get_async_session() as session:
            query = select(func.count(self.model_class.id))
            
            # Apply soft delete filter
            if not include_deleted and hasattr(self.model_class, 'is_deleted'):
                query = query.where(self.model_class.is_deleted == False)
            
            # Apply filters
            for filter_spec in filters:
                query = filter_spec.apply(query, self.model_class)
            
            result = await session.execute(query)
            return result.scalar()
    
    async def exists(self, filters: List[QueryFilter]) -> bool:
        """Check if any entity exists matching filters."""
        count = await self.count_by_filters(filters)
        return count > 0
    
    # Batch operations
    
    async def create_many(self, data_list: List[Union[Dict[str, Any], CreateSchemaType]]) -> List[T]:
        """Create multiple entities in a single transaction."""
        if not data_list:
            return []
        
        # Convert to dict format
        clean_data = []
        for data in data_list:
            if not isinstance(data, dict):
                data = data.dict() if hasattr(data, 'dict') else data.__dict__
            clean_data.append(data)
        
        async with db_manager.get_async_session() as session:
            entities = []
            for data in clean_data:
                entity = self.model_class(**data)
                session.add(entity)
                entities.append(entity)
            
            await session.commit()
            
            # Refresh all entities
            for entity in entities:
                await session.refresh(entity)
            
            # Invalidate caches
            await self._invalidate_cache_pattern(f"{self.model_name}:*")
            
            return entities
    
    async def update_many(self, updates: List[Dict[str, Any]]) -> int:
        """Update multiple entities. Each update dict should contain 'id' and update fields."""
        if not updates:
            return 0
        
        async with db_manager.get_async_session() as session:
            updated_count = 0
            
            for update_data in updates:
                entity_id = update_data.pop('id')
                clean_data = {k: v for k, v in update_data.items() if v is not None}
                clean_data['updated_at'] = datetime.utcnow()
                
                result = await session.execute(
                    update(self.model_class)
                    .where(self.model_class.id == entity_id)
                    .values(**clean_data)
                )
                
                if result.rowcount > 0:
                    updated_count += 1
            
            await session.commit()
            
            # Invalidate caches
            await self._invalidate_cache_pattern(f"{self.model_name}:*")
            
            return updated_count
    
    async def delete_by_filters(self, filters: List[QueryFilter], soft_delete: bool = True) -> int:
        """Delete entities matching filters."""
        if not filters:
            raise ValueError("Filters required for bulk delete operations")
        
        async with db_manager.get_async_session() as session:
            if soft_delete and hasattr(self.model_class, 'is_deleted'):
                # Soft delete
                query = update(self.model_class).values(
                    is_deleted=True,
                    deleted_at=datetime.utcnow()
                )
            else:
                # Hard delete
                query = delete(self.model_class)
            
            # Apply filters
            for filter_spec in filters:
                query = filter_spec.apply(query, self.model_class)
            
            result = await session.execute(query)
            await session.commit()
            
            # Invalidate caches
            await self._invalidate_cache_pattern(f"{self.model_name}:*")
            
            return result.rowcount
    
    # Analytics and aggregation
    
    async def get_field_stats(self, field: str, filters: List[QueryFilter] = None) -> Dict[str, Any]:
        """Get statistical information about a numeric field."""
        filters = filters or []
        
        async with db_manager.get_async_session() as session:
            field_attr = getattr(self.model_class, field)
            
            query = select([
                func.count(field_attr).label('count'),
                func.sum(field_attr).label('sum'),
                func.avg(field_attr).label('avg'),
                func.min(field_attr).label('min'),
                func.max(field_attr).label('max'),
                func.stddev(field_attr).label('stddev')
            ])
            
            # Apply filters
            for filter_spec in filters:
                query = filter_spec.apply(query, self.model_class)
            
            result = await session.execute(query)
            row = result.first()
            
            return {
                'count': row.count or 0,
                'sum': float(row.sum) if row.sum else 0.0,
                'avg': float(row.avg) if row.avg else 0.0,
                'min': float(row.min) if row.min else 0.0,
                'max': float(row.max) if row.max else 0.0,
                'stddev': float(row.stddev) if row.stddev else 0.0,
            }
    
    async def group_by_field(
        self,
        group_field: str,
        agg_field: str = None,
        agg_func: str = "count",
        filters: List[QueryFilter] = None
    ) -> List[Dict[str, Any]]:
        """Group entities by a field and apply aggregation."""
        filters = filters or []
        
        async with db_manager.get_async_session() as session:
            group_attr = getattr(self.model_class, group_field)
            
            if agg_field:
                agg_attr = getattr(self.model_class, agg_field)
                if agg_func == "sum":
                    agg_expr = func.sum(agg_attr)
                elif agg_func == "avg":
                    agg_expr = func.avg(agg_attr)
                elif agg_func == "min":
                    agg_expr = func.min(agg_attr)
                elif agg_func == "max":
                    agg_expr = func.max(agg_attr)
                else:
                    agg_expr = func.count(agg_attr)
            else:
                agg_expr = func.count(self.model_class.id)
            
            query = (
                select([group_attr.label('group_value'), agg_expr.label('agg_value')])
                .group_by(group_attr)
                .order_by(desc(agg_expr))
            )
            
            # Apply filters
            for filter_spec in filters:
                query = filter_spec.apply(query, self.model_class)
            
            result = await session.execute(query)
            
            return [
                {
                    'group_value': row.group_value,
                    'agg_value': float(row.agg_value) if row.agg_value else 0
                }
                for row in result
            ]
    
    # Health and monitoring
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on repository."""
        start_time = datetime.utcnow()
        
        try:
            # Test basic query
            async with db_manager.get_async_session() as session:
                await session.execute(select(func.count(self.model_class.id)))
                
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy',
                'model': self.model_name,
                'response_time_ms': response_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                'status': 'unhealthy',
                'model': self.model_name,
                'error': str(e),
                'response_time_ms': response_time,
                'timestamp': datetime.utcnow().isoformat()
            }


class TransactionManager:
    """Transaction manager for complex operations across multiple repositories."""
    
    def __init__(self):
        self._session: Optional[AsyncSession] = None
    
    @asynccontextmanager
    async def transaction(self):
        """Async context manager for database transactions."""
        async with db_manager.get_async_session() as session:
            self._session = session
            try:
                yield self
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                self._session = None
    
    async def execute_in_transaction(self, func: Callable, *args, **kwargs):
        """Execute a function within the current transaction."""
        if not self._session:
            raise RuntimeError("No active transaction")
        
        return await func(self._session, *args, **kwargs)


# Global transaction manager instance
transaction_manager = TransactionManager()