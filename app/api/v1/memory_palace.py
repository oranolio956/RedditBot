"""
Memory Palace API Endpoints

Provides RESTful API for spatial memory navigation and management.
Implements method of loci-based conversation storage and retrieval.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import json

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, func
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis

from app.database.session import get_db
from app.core.redis import get_redis
from app.models.user import User
from app.models.memory_palace import (
    MemoryPalace, MemoryRoom, SpatialMemory,
    NavigationPath, MemoryAnchor
)
from app.models.message import Message
from app.services.memory_palace import MemoryPalaceEngine
from app.api.deps import get_current_user
from app.core.security import verify_jwt_token
from app.core.security_utils import RateLimiter, ConsentManager
from app.config.settings import get_settings
import jwt

router = APIRouter()
security = HTTPBearer()
rate_limiter = RateLimiter()
consent_manager = ConsentManager()


# JWT verification is now handled by the centralized security module


# Request/Response Models
class CreatePalaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    layout_type: str = Field(default="mansion", regex="^(mansion|castle|modern|temple|custom)$")
    floor_count: int = Field(default=3, ge=1, le=10)
    
    
class PalaceResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    layout_type: str
    room_count: int
    total_memories: int
    memory_density: float
    retrieval_success_rate: float
    navigation_efficiency: float
    created_at: datetime
    
    class Config:
        orm_mode = True
        

class StoreMemoryRequest(BaseModel):
    message_id: UUID
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion: float = Field(default=0.0, ge=-1.0, le=1.0)
    category: Optional[str] = None
    associations: List[str] = Field(default_factory=list)
    

class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    use_spatial: bool = Field(default=True, description="Consider spatial proximity")
    

class NavigationRequest(BaseModel):
    target_type: str = Field(..., regex="^(memory|room|position)$")
    target_id: Optional[UUID] = None
    target_position: Optional[List[float]] = None
    query: Optional[str] = None
    

class MemoryTrailRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    max_memories: int = Field(default=20, ge=1, le=50)
    optimize_path: bool = Field(default=True)
    

class RoomResponse(BaseModel):
    id: UUID
    name: str
    theme: str
    floor_level: int
    position_3d: List[float]
    dimensions_3d: List[float]
    current_occupancy: int
    memory_capacity: int
    emotional_tone: float
    visit_count: int
    last_visited: Optional[datetime]
    
    class Config:
        orm_mode = True
        

class SpatialMemoryResponse(BaseModel):
    id: UUID
    room_id: UUID
    position_3d: List[float]
    memory_strength: float
    emotional_valence: float
    semantic_category: Optional[str]
    content_summary: Optional[str]
    keywords: List[str]
    visual_symbol: Optional[str]
    access_count: int
    last_accessed: Optional[datetime]
    message_preview: Optional[str] = None
    
    class Config:
        orm_mode = True


# Palace Management Endpoints
@router.post("/palaces", response_model=PalaceResponse)
async def create_palace(
    request: CreatePalaceRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Create a new memory palace for the user."""
    engine = MemoryPalaceEngine(db, redis_client)
    
    palace = await engine.create_palace(
        user_id=current_user.id,
        name=request.name,
        layout_type=request.layout_type
    )
    
    # Set floor count if different from default
    if request.floor_count != 3:
        palace.floor_count = request.floor_count
        palace.spatial_configuration["floors"] = request.floor_count
        await db.commit()
        await db.refresh(palace)
    
    if request.description:
        palace.description = request.description
        await db.commit()
        await db.refresh(palace)
    
    return PalaceResponse.from_orm(palace)


@router.get("/palaces", response_model=List[PalaceResponse])
async def list_palaces(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all memory palaces for the current user."""
    result = await db.execute(
        select(MemoryPalace)
        .where(MemoryPalace.user_id == current_user.id)
        .order_by(MemoryPalace.created_at.desc())
    )
    
    palaces = result.scalars().all()
    return [PalaceResponse.from_orm(p) for p in palaces]


@router.get("/palaces/{palace_id}", response_model=PalaceResponse)
async def get_palace(
    palace_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific memory palace."""
    result = await db.execute(
        select(MemoryPalace)
        .where(
            and_(
                MemoryPalace.id == palace_id,
                MemoryPalace.user_id == current_user.id
            )
        )
    )
    
    palace = result.scalar_one_or_none()
    if not palace:
        raise HTTPException(status_code=404, detail="Memory palace not found")
    
    return PalaceResponse.from_orm(palace)


# Room Management Endpoints
@router.get("/palaces/{palace_id}/rooms", response_model=List[RoomResponse])
async def list_rooms(
    palace_id: UUID,
    floor: Optional[int] = Query(None, description="Filter by floor level"),
    theme: Optional[str] = Query(None, description="Filter by theme"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all rooms in a memory palace."""
    # Verify palace ownership
    result = await db.execute(
        select(MemoryPalace)
        .where(
            and_(
                MemoryPalace.id == palace_id,
                MemoryPalace.user_id == current_user.id
            )
        )
    )
    
    palace = result.scalar_one_or_none()
    if not palace:
        raise HTTPException(status_code=404, detail="Memory palace not found")
    
    # Build room query
    query = select(MemoryRoom).where(MemoryRoom.palace_id == palace_id)
    
    if floor is not None:
        query = query.where(MemoryRoom.floor_level == floor)
    
    if theme:
        query = query.where(MemoryRoom.theme.ilike(f"%{theme}%"))
    
    query = query.order_by(MemoryRoom.floor_level, MemoryRoom.name)
    
    result = await db.execute(query)
    rooms = result.scalars().all()
    
    return [RoomResponse.from_orm(r) for r in rooms]


@router.get("/palaces/{palace_id}/rooms/{room_id}", response_model=RoomResponse)
async def get_room(
    palace_id: UUID,
    room_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific room."""
    result = await db.execute(
        select(MemoryRoom)
        .join(MemoryPalace)
        .where(
            and_(
                MemoryRoom.id == room_id,
                MemoryRoom.palace_id == palace_id,
                MemoryPalace.user_id == current_user.id
            )
        )
    )
    
    room = result.scalar_one_or_none()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    # Update visit statistics
    room.visit_count += 1
    room.last_visited = datetime.utcnow()
    await db.commit()
    await db.refresh(room)
    
    return RoomResponse.from_orm(room)


# Memory Storage and Retrieval
@router.post("/memories/store", response_model=SpatialMemoryResponse)
async def store_memory(
    request: StoreMemoryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Store a conversation message in the memory palace."""
    # Get the message
    result = await db.execute(
        select(Message)
        .where(
            and_(
                Message.id == request.message_id,
                Message.user_id == current_user.id
            )
        )
    )
    
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Prepare context
    context = {
        "importance": request.importance,
        "emotion": request.emotion,
        "category": request.category,
        "associations": request.associations
    }
    
    # Store in memory palace
    engine = MemoryPalaceEngine(db, redis_client)
    spatial_memory = await engine.store_conversation(
        user_id=current_user.id,
        message=message,
        context=context
    )
    
    response = SpatialMemoryResponse.from_orm(spatial_memory)
    response.message_preview = message.content[:200] if message.content else None
    
    return response


@router.post("/memories/search", response_model=List[SpatialMemoryResponse])
async def search_memories(
    request: MemorySearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Search for memories using semantic and spatial criteria."""
    engine = MemoryPalaceEngine(db, redis_client)
    
    memories = await engine.search_memories(
        user_id=current_user.id,
        query=request.query,
        limit=request.limit
    )
    
    responses = []
    for memory in memories:
        response = SpatialMemoryResponse.from_orm(memory)
        if memory.message:
            response.message_preview = memory.message.content[:200]
        responses.append(response)
    
    return responses


@router.get("/memories/nearby", response_model=List[SpatialMemoryResponse])
async def get_nearby_memories(
    radius: float = Query(10.0, ge=1.0, le=100.0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Get memories near the current navigation position."""
    engine = MemoryPalaceEngine(db, redis_client)
    
    memories = await engine.get_nearby_memories(
        user_id=current_user.id,
        radius=radius
    )
    
    responses = []
    for memory in memories:
        response = SpatialMemoryResponse.from_orm(memory)
        if memory.message:
            response.message_preview = memory.message.content[:200]
        responses.append(response)
    
    return responses


# Navigation Endpoints
@router.post("/navigate")
async def navigate_to_target(
    request: NavigationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Navigate to a specific target in the memory palace."""
    engine = MemoryPalaceEngine(db, redis_client)
    
    if request.target_type == "memory" and request.query:
        # Navigate to memory by search
        memory, path = await engine.navigate_to_memory(
            user_id=current_user.id,
            query=request.query
        )
        
        if not memory:
            raise HTTPException(status_code=404, detail="No matching memory found")
        
        return {
            "success": True,
            "target_type": "memory",
            "target_id": str(memory.id),
            "position": memory.position_3d,
            "room_id": str(memory.room_id),
            "path_distance": path.total_distance if path else 0,
            "navigation_time": path.navigation_time if path else 0
        }
    
    elif request.target_type == "room" and request.target_id:
        # Navigate to specific room
        result = await db.execute(
            select(MemoryRoom)
            .join(MemoryPalace)
            .where(
                and_(
                    MemoryRoom.id == request.target_id,
                    MemoryPalace.user_id == current_user.id
                )
            )
        )
        
        room = result.scalar_one_or_none()
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Update navigation
        from app.services.memory_palace import MemoryRoom3D
        room_3d = MemoryRoom3D(
            room_id=str(room.id),
            theme=room.theme,
            position=tuple(room.position_3d)
        )
        
        await engine.navigation.move_to_room(room_3d)
        
        return {
            "success": True,
            "target_type": "room",
            "target_id": str(room.id),
            "position": room.position_3d,
            "room_name": room.name,
            "room_theme": room.theme
        }
    
    elif request.target_type == "position" and request.target_position:
        # Navigate to specific 3D position
        if len(request.target_position) != 3:
            raise HTTPException(status_code=400, detail="Position must be [x, y, z]")
        
        engine.navigation.current_position = tuple(request.target_position)
        
        return {
            "success": True,
            "target_type": "position",
            "position": request.target_position
        }
    
    else:
        raise HTTPException(status_code=400, detail="Invalid navigation request")


@router.post("/memories/trail", response_model=List[SpatialMemoryResponse])
async def create_memory_trail(
    request: MemoryTrailRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Create a guided trail through related memories."""
    engine = MemoryPalaceEngine(db, redis_client)
    
    memories = await engine.create_memory_trail(
        user_id=current_user.id,
        topic=request.topic
    )
    
    # Limit to requested max
    memories = memories[:request.max_memories]
    
    responses = []
    for memory in memories:
        response = SpatialMemoryResponse.from_orm(memory)
        if memory.message:
            response.message_preview = memory.message.content[:200]
        responses.append(response)
    
    return responses


# 3D Visualization Export
@router.get("/palaces/{palace_id}/export3d")
async def export_palace_3d(
    palace_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Export palace data for 3D visualization."""
    # Verify ownership
    result = await db.execute(
        select(MemoryPalace)
        .where(
            and_(
                MemoryPalace.id == palace_id,
                MemoryPalace.user_id == current_user.id
            )
        )
    )
    
    palace = result.scalar_one_or_none()
    if not palace:
        raise HTTPException(status_code=404, detail="Memory palace not found")
    
    engine = MemoryPalaceEngine(db, redis_client)
    scene_data = await engine.export_palace_data(current_user.id)
    
    return JSONResponse(content=scene_data)


# WebSocket for Real-time Navigation
@router.websocket("/palaces/{palace_id}/navigate")
async def websocket_navigation(
    websocket: WebSocket,
    palace_id: UUID,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """WebSocket endpoint for real-time 3D navigation."""
    # Verify token
    try:
        payload = verify_token(token)
        user_id = UUID(payload.get("sub"))
    except Exception:
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    # Verify palace ownership
    result = await db.execute(
        select(MemoryPalace)
        .where(
            and_(
                MemoryPalace.id == palace_id,
                MemoryPalace.user_id == user_id
            )
        )
    )
    
    palace = result.scalar_one_or_none()
    if not palace:
        await websocket.close(code=4004, reason="Palace not found")
        return
    
    await websocket.accept()
    
    engine = MemoryPalaceEngine(db, redis_client)
    
    try:
        while True:
            # Receive navigation commands
            data = await websocket.receive_json()
            
            command = data.get("command")
            
            if command == "move":
                position = data.get("position", [0, 0, 0])
                engine.navigation.current_position = tuple(position)
                
                # Get nearby memories
                memories = await engine.get_nearby_memories(user_id, radius=15.0)
                
                await websocket.send_json({
                    "type": "position_update",
                    "position": position,
                    "nearby_memories": len(memories),
                    "memories": [
                        {
                            "id": str(m.id),
                            "position": m.position_3d,
                            "strength": m.memory_strength,
                            "category": m.semantic_category
                        }
                        for m in memories[:5]  # Send top 5
                    ]
                })
            
            elif command == "look":
                direction = data.get("direction", [0, 0, 1])
                
                # Could implement ray casting for what user sees
                await websocket.send_json({
                    "type": "view_update",
                    "direction": direction
                })
            
            elif command == "teleport":
                room_id = data.get("room_id")
                if room_id:
                    result = await db.execute(
                        select(MemoryRoom)
                        .where(
                            and_(
                                MemoryRoom.id == UUID(room_id),
                                MemoryRoom.palace_id == palace_id
                            )
                        )
                    )
                    
                    room = result.scalar_one_or_none()
                    if room:
                        engine.navigation.current_position = tuple(room.position_3d)
                        
                        await websocket.send_json({
                            "type": "teleport_complete",
                            "room_id": room_id,
                            "position": room.position_3d,
                            "room_name": room.name
                        })
            
            elif command == "get_scene":
                # Send full scene data
                scene = await engine.export_palace_data(user_id)
                await websocket.send_json({
                    "type": "scene_data",
                    "data": scene
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))


# Memory Reinforcement
@router.post("/memories/{memory_id}/reinforce")
async def reinforce_memory(
    memory_id: UUID,
    strength_boost: float = Query(0.1, ge=0.0, le=0.5),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Reinforce a memory to prevent decay."""
    result = await db.execute(
        select(SpatialMemory)
        .join(MemoryPalace)
        .where(
            and_(
                SpatialMemory.id == memory_id,
                MemoryPalace.user_id == current_user.id
            )
        )
    )
    
    memory = result.scalar_one_or_none()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Reinforce memory
    memory.memory_strength = min(1.0, memory.memory_strength + strength_boost)
    memory.reinforcement_count += 1
    memory.last_accessed = datetime.utcnow()
    memory.access_count += 1
    
    # Update palace statistics
    palace = await db.get(MemoryPalace, memory.palace_id)
    if palace:
        palace.retrieval_success_rate = (
            (palace.retrieval_success_rate * palace.total_memories + 1) /
            (palace.total_memories + 1)
        )
    
    await db.commit()
    
    return {
        "success": True,
        "memory_id": str(memory_id),
        "new_strength": memory.memory_strength,
        "reinforcement_count": memory.reinforcement_count
    }


# Analytics Endpoints
@router.get("/palaces/{palace_id}/analytics")
async def get_palace_analytics(
    palace_id: UUID,
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get analytics for memory palace usage."""
    # Verify ownership
    result = await db.execute(
        select(MemoryPalace)
        .where(
            and_(
                MemoryPalace.id == palace_id,
                MemoryPalace.user_id == current_user.id
            )
        )
    )
    
    palace = result.scalar_one_or_none()
    if not palace:
        raise HTTPException(status_code=404, detail="Memory palace not found")
    
    since = datetime.utcnow() - timedelta(days=days)
    
    # Get memory statistics
    result = await db.execute(
        select(
            func.count(SpatialMemory.id).label("total_memories"),
            func.avg(SpatialMemory.memory_strength).label("avg_strength"),
            func.avg(SpatialMemory.access_count).label("avg_access_count"),
            func.count(func.distinct(SpatialMemory.semantic_category)).label("unique_categories")
        )
        .where(
            and_(
                SpatialMemory.palace_id == palace_id,
                SpatialMemory.created_at >= since
            )
        )
    )
    
    memory_stats = result.one()
    
    # Get room statistics
    result = await db.execute(
        select(
            func.count(MemoryRoom.id).label("total_rooms"),
            func.avg(MemoryRoom.current_occupancy).label("avg_occupancy"),
            func.sum(MemoryRoom.visit_count).label("total_visits"),
            func.max(MemoryRoom.visit_count).label("max_visits")
        )
        .where(MemoryRoom.palace_id == palace_id)
    )
    
    room_stats = result.one()
    
    # Get navigation statistics
    result = await db.execute(
        select(
            func.count(NavigationPath.id).label("total_navigations"),
            func.avg(NavigationPath.total_distance).label("avg_distance"),
            func.avg(NavigationPath.navigation_time).label("avg_time"),
            func.avg(NavigationPath.efficiency_score).label("avg_efficiency")
        )
        .where(
            and_(
                NavigationPath.palace_id == palace_id,
                NavigationPath.created_at >= since
            )
        )
    )
    
    nav_stats = result.one()
    
    # Get most visited rooms
    result = await db.execute(
        select(MemoryRoom)
        .where(MemoryRoom.palace_id == palace_id)
        .order_by(MemoryRoom.visit_count.desc())
        .limit(5)
    )
    
    top_rooms = result.scalars().all()
    
    return {
        "palace_id": str(palace_id),
        "period_days": days,
        "memory_statistics": {
            "total_memories": memory_stats.total_memories or 0,
            "average_strength": float(memory_stats.avg_strength or 0),
            "average_access_count": float(memory_stats.avg_access_count or 0),
            "unique_categories": memory_stats.unique_categories or 0
        },
        "room_statistics": {
            "total_rooms": room_stats.total_rooms or 0,
            "average_occupancy": float(room_stats.avg_occupancy or 0),
            "total_visits": room_stats.total_visits or 0,
            "most_visited_count": room_stats.max_visits or 0
        },
        "navigation_statistics": {
            "total_navigations": nav_stats.total_navigations or 0,
            "average_distance": float(nav_stats.avg_distance or 0),
            "average_time": float(nav_stats.avg_time or 0),
            "average_efficiency": float(nav_stats.avg_efficiency or 0)
        },
        "top_rooms": [
            {
                "id": str(room.id),
                "name": room.name,
                "theme": room.theme,
                "visit_count": room.visit_count
            }
            for room in top_rooms
        ]
    }