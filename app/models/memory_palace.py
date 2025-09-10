"""
Memory Palace Database Models

Stores spatial memory structures, navigation paths, and 3D palace configurations.
Based on method of loci research showing 8.8% recall improvement.
"""

from datetime import datetime
from typing import Optional, Dict, List, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text,
    ForeignKey, JSON, Index, DateTime, ARRAY, CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

from app.database.base import Base, FullAuditModel


class MemoryPalace(FullAuditModel):
    """
    Represents a user's memory palace structure.
    Each user can have multiple palaces for different purposes.
    """
    
    __tablename__ = "memory_palaces"
    
    # User relationship
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="memory_palaces")
    
    # Palace metadata
    name = Column(String(200), nullable=False)
    description = Column(Text)
    layout_type = Column(
        String(50),
        nullable=False,
        default="mansion",
        comment="mansion, castle, modern, temple, custom"
    )
    
    # Spatial configuration
    room_count = Column(Integer, default=0, nullable=False)
    floor_count = Column(Integer, default=3, nullable=False)
    total_volume = Column(Float, default=1000.0, comment="Total 3D volume in cubic units")
    
    # Memory statistics
    total_memories = Column(Integer, default=0, nullable=False)
    memory_density = Column(Float, default=0.0, comment="Memories per cubic unit")
    retrieval_success_rate = Column(Float, default=0.0, comment="Successful recalls / attempts")
    
    # Navigation metrics
    navigation_efficiency = Column(Float, default=1.0, comment="Path optimality score")
    average_retrieval_time = Column(Float, comment="Avg seconds to find memory")
    most_visited_room_id = Column(PG_UUID(as_uuid=True))
    
    # 3D configuration (JSONB for complex scene data)
    spatial_configuration = Column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Complete 3D scene configuration"
    )
    
    # Visual theme
    visual_theme = Column(
        JSONB,
        default=dict,
        comment="Colors, textures, lighting settings"
    )
    
    # Performance optimization
    spatial_index_data = Column(
        JSONB,
        comment="R-tree spatial index cache"
    )
    
    # Relationships
    rooms = relationship("MemoryRoom", back_populates="palace", cascade="all, delete-orphan")
    spatial_memories = relationship("SpatialMemory", back_populates="palace", cascade="all, delete-orphan")
    navigation_paths = relationship("NavigationPath", back_populates="palace", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_palace_user_id', 'user_id'),
        Index('idx_palace_retrieval_rate', 'retrieval_success_rate'),
        CheckConstraint('room_count >= 0', name='check_room_count_positive'),
        CheckConstraint('memory_density >= 0', name='check_density_positive'),
    )


class MemoryRoom(FullAuditModel):
    """
    Individual room within a memory palace.
    Each room has a theme and contains memory anchors.
    """
    
    __tablename__ = "memory_rooms"
    
    # Palace relationship
    palace_id = Column(PG_UUID(as_uuid=True), ForeignKey("memory_palaces.id"), nullable=False)
    palace = relationship("MemoryPalace", back_populates="rooms")
    
    # Room properties
    name = Column(String(200), nullable=False)
    theme = Column(String(100), nullable=False, comment="Room theme for associations")
    floor_level = Column(Integer, default=0, nullable=False)
    
    # 3D position and dimensions
    position_3d = Column(
        ARRAY(Float, dimensions=1),
        nullable=False,
        comment="[x, y, z] position in palace"
    )
    dimensions_3d = Column(
        ARRAY(Float, dimensions=1),
        default=[10.0, 8.0, 10.0],
        comment="[width, height, depth]"
    )
    
    # Memory capacity
    anchor_points = Column(
        JSONB,
        default=list,
        nullable=False,
        comment="Available positions for memories"
    )
    memory_capacity = Column(Integer, default=10, nullable=False)
    current_occupancy = Column(Integer, default=0, nullable=False)
    
    # Room characteristics
    emotional_tone = Column(Float, default=0.5, comment="0=negative, 1=positive")
    lighting_level = Column(Float, default=0.7, comment="0=dark, 1=bright")
    temperature_feel = Column(Float, default=0.5, comment="0=cold, 1=warm")
    
    # Navigation data
    connected_rooms = Column(
        ARRAY(PG_UUID(as_uuid=True), dimensions=1),
        default=list,
        comment="IDs of connected rooms"
    )
    visit_count = Column(Integer, default=0, nullable=False)
    last_visited = Column(DateTime)
    average_dwell_time = Column(Float, comment="Avg seconds spent in room")
    
    # Visual configuration
    visual_markers = Column(
        JSONB,
        default=dict,
        comment="Visual elements and decorations"
    )
    
    # Relationships
    spatial_memories = relationship("SpatialMemory", back_populates="room", cascade="all, delete-orphan")
    memory_anchors = relationship("MemoryAnchor", back_populates="room", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_room_palace_theme', 'palace_id', 'theme'),
        Index('idx_room_floor', 'floor_level'),
        Index('idx_room_occupancy', 'current_occupancy'),
        CheckConstraint('current_occupancy <= memory_capacity', name='check_occupancy_limit'),
    )


class SpatialMemory(FullAuditModel):
    """
    A memory stored at a specific location in the palace.
    Links conversations to spatial positions.
    """
    
    __tablename__ = "spatial_memories"
    
    # Palace and room relationship
    palace_id = Column(PG_UUID(as_uuid=True), ForeignKey("memory_palaces.id"), nullable=False)
    palace = relationship("MemoryPalace", back_populates="spatial_memories")
    
    room_id = Column(PG_UUID(as_uuid=True), ForeignKey("memory_rooms.id"), nullable=False)
    room = relationship("MemoryRoom", back_populates="spatial_memories")
    
    # Message relationship
    message_id = Column(PG_UUID(as_uuid=True), ForeignKey("messages.id"))
    message = relationship("Message", back_populates="spatial_memory")
    
    # 3D position within room
    position_3d = Column(
        ARRAY(Float, dimensions=1),
        nullable=False,
        comment="[x, y, z] position in room"
    )
    
    # Memory properties
    memory_strength = Column(Float, default=1.0, nullable=False, comment="0=weak, 1=strong")
    decay_rate = Column(Float, default=0.01, comment="Strength decay per day")
    reinforcement_count = Column(Integer, default=0, nullable=False)
    
    # Emotional and semantic data
    emotional_valence = Column(Float, default=0.0, comment="-1=negative, 1=positive")
    emotional_arousal = Column(Float, default=0.0, comment="0=calm, 1=excited")
    semantic_category = Column(String(100), comment="Primary category")
    
    # Memory content summary
    content_summary = Column(Text, comment="Brief summary for quick recall")
    keywords = Column(ARRAY(String), default=list, comment="Searchable keywords")
    
    # Visual representation
    visual_symbol = Column(String(100), comment="Icon or 3D model identifier")
    color_encoding = Column(ARRAY(Float), comment="[R, G, B] color values")
    
    # Associations
    associations = Column(
        JSONB,
        default=dict,
        comment="Links to related memories and concepts"
    )
    
    # Memory vector for similarity search
    memory_vector = Column(
        ARRAY(Float, dimensions=1),
        comment="Embedding vector for semantic search"
    )
    
    # Access patterns
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed = Column(DateTime)
    average_recall_time = Column(Float, comment="Avg seconds to recall")
    
    # Spatial index for efficient queries
    spatial_hash = Column(String(64), comment="Geohash for spatial indexing")
    
    # Indexes
    __table_args__ = (
        Index('idx_spatial_memory_palace_room', 'palace_id', 'room_id'),
        Index('idx_spatial_memory_strength', 'memory_strength'),
        Index('idx_spatial_memory_category', 'semantic_category'),
        Index('idx_spatial_memory_hash', 'spatial_hash'),
        Index('idx_spatial_memory_message', 'message_id'),
    )


class MemoryAnchor(FullAuditModel):
    """
    Specific anchor points within rooms for memory placement.
    Based on method of loci anchor point research.
    """
    
    __tablename__ = "memory_anchors"
    
    # Room relationship
    room_id = Column(PG_UUID(as_uuid=True), ForeignKey("memory_rooms.id"), nullable=False)
    room = relationship("MemoryRoom", back_populates="memory_anchors")
    
    # Anchor properties
    name = Column(String(200), nullable=False)
    anchor_type = Column(
        String(50),
        nullable=False,
        comment="corner, wall, floor, ceiling, furniture, object"
    )
    
    # 3D position
    position_3d = Column(
        ARRAY(Float, dimensions=1),
        nullable=False,
        comment="[x, y, z] relative to room origin"
    )
    
    # Visual marker
    visual_marker = Column(
        String(100),
        nullable=False,
        comment="3D model or icon identifier"
    )
    marker_size = Column(Float, default=1.0, comment="Scale factor")
    marker_color = Column(ARRAY(Float), comment="[R, G, B, A] color")
    
    # Memory association
    occupied = Column(Boolean, default=False, nullable=False)
    memory_id = Column(PG_UUID(as_uuid=True), ForeignKey("spatial_memories.id"))
    association_strength = Column(Float, default=0.0, comment="How well memory fits anchor")
    
    # Anchor characteristics
    memorability_score = Column(Float, default=0.5, comment="How memorable this anchor is")
    uniqueness_score = Column(Float, default=0.5, comment="How unique within room")
    
    # Usage statistics
    use_count = Column(Integer, default=0, nullable=False)
    success_rate = Column(Float, default=0.0, comment="Successful recalls / attempts")
    
    # Indexes
    __table_args__ = (
        Index('idx_anchor_room_occupied', 'room_id', 'occupied'),
        Index('idx_anchor_memory', 'memory_id'),
    )


class NavigationPath(FullAuditModel):
    """
    Records navigation paths through the memory palace.
    Used for optimizing retrieval and understanding usage patterns.
    """
    
    __tablename__ = "navigation_paths"
    
    # Palace relationship
    palace_id = Column(PG_UUID(as_uuid=True), ForeignKey("memory_palaces.id"), nullable=False)
    palace = relationship("MemoryPalace", back_populates="navigation_paths")
    
    # User relationship
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Path data
    start_position = Column(
        ARRAY(Float, dimensions=1),
        nullable=False,
        comment="[x, y, z] starting position"
    )
    end_position = Column(
        ARRAY(Float, dimensions=1),
        nullable=False,
        comment="[x, y, z] ending position"
    )
    
    # Waypoints
    waypoints = Column(
        JSONB,
        default=list,
        comment="List of [x, y, z] positions along path"
    )
    
    # Path metrics
    total_distance = Column(Float, nullable=False, comment="Total distance traveled")
    navigation_time = Column(Float, nullable=False, comment="Time taken in seconds")
    efficiency_score = Column(Float, comment="Optimal distance / actual distance")
    
    # Navigation purpose
    navigation_type = Column(
        String(50),
        comment="search, recall, explore, guided"
    )
    target_memory_id = Column(PG_UUID(as_uuid=True), ForeignKey("spatial_memories.id"))
    found_target = Column(Boolean, default=False)
    
    # Rooms visited
    rooms_visited = Column(
        ARRAY(PG_UUID(as_uuid=True), dimensions=1),
        default=list,
        comment="Ordered list of room IDs"
    )
    
    # User behavior
    pause_locations = Column(
        JSONB,
        default=list,
        comment="Positions where user paused"
    )
    backtrack_count = Column(Integer, default=0, comment="Times user went backwards")
    
    # Indexes
    __table_args__ = (
        Index('idx_nav_palace_user', 'palace_id', 'user_id'),
        Index('idx_nav_type', 'navigation_type'),
        Index('idx_nav_efficiency', 'efficiency_score'),
    )


class SpatialIndex(FullAuditModel):
    """
    Spatial index cache for fast 3D queries.
    Stores R-tree nodes for efficient spatial searches.
    """
    
    __tablename__ = "spatial_indices"
    
    # Palace relationship
    palace_id = Column(PG_UUID(as_uuid=True), ForeignKey("memory_palaces.id"), nullable=False, unique=True)
    
    # Index structure (R-tree nodes)
    index_data = Column(
        JSONB,
        nullable=False,
        comment="Serialized R-tree structure"
    )
    
    # Index statistics
    node_count = Column(Integer, default=0, nullable=False)
    depth = Column(Integer, default=0, nullable=False)
    item_count = Column(Integer, default=0, nullable=False)
    
    # Performance metrics
    average_query_time = Column(Float, comment="Avg query time in ms")
    cache_hit_rate = Column(Float, default=0.0, comment="Cache hit ratio")
    
    # Rebuild tracking
    last_rebuild = Column(DateTime, default=datetime.utcnow, nullable=False)
    rebuild_count = Column(Integer, default=0, nullable=False)
    needs_rebuild = Column(Boolean, default=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_spatial_index_palace', 'palace_id'),
        Index('idx_spatial_index_rebuild', 'needs_rebuild'),
    )