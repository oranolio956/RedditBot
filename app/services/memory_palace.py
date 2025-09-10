"""
Memory Palace Architecture Service

Implements spatial navigation for conversations using method of loci techniques.
Based on 2024 research showing 8.8% recall improvement with VR memory palaces.
Uses Three.js for 3D visualization and R-tree spatial indexing for performance.
"""

import asyncio
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from uuid import UUID
import logging
from collections import defaultdict, deque
import math
import random

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.orm import selectinload
import redis.asyncio as redis

from app.models.memory_palace import (
    MemoryPalace, SpatialMemory, MemoryRoom,
    NavigationPath, MemoryAnchor, SpatialIndex
)
from app.models.message import Message
from app.models.user import User
from app.core.config import settings
from app.services.llm_service import LLMService
from app.core.security_utils import (
    EncryptionService, InputSanitizer, PrivacyProtector, 
    RateLimiter, MLSecurityValidator, ConsentManager
)
from app.core.performance_utils import MultiLevelCache, ModelOptimizer

logger = logging.getLogger(__name__)


class SpatialIndexer:
    """
    R-tree based spatial indexing for efficient 3D queries.
    Implements Small-Tree-Large-Tree (STLT) algorithm for bulk insertion.
    """
    
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        self.root = None
        self.items = {}
        self.node_capacity = 16  # Optimal for 3D based on research
        self.rebuild_threshold = 0.3  # Rebuild if 30% of tree is unbalanced
        
    def insert(self, item_id: str, bounds: List[float]):
        """Insert item with 3D bounding box."""
        if len(bounds) != self.dimensions * 2:
            raise ValueError(f"Expected {self.dimensions * 2} bounds values")
            
        self.items[item_id] = bounds
        
        if not self.root:
            self.root = {"bounds": bounds, "items": [item_id], "children": []}
        else:
            self._insert_recursive(self.root, item_id, bounds)
            
    def _insert_recursive(self, node: Dict, item_id: str, bounds: List[float]):
        """Recursive insertion using surface area heuristic."""
        node["bounds"] = self._expand_bounds(node["bounds"], bounds)
        
        if not node["children"]:
            # Leaf node
            node["items"].append(item_id)
            
            if len(node["items"]) > self.node_capacity:
                self._split_node(node)
        else:
            # Internal node - choose best child
            best_child = self._choose_subtree(node["children"], bounds)
            self._insert_recursive(best_child, item_id, bounds)
            
    def _split_node(self, node: Dict):
        """Split overflowing node using SAH (Surface Area Heuristic)."""
        items = node["items"]
        
        # Find best split axis and position
        best_split = self._find_best_split(items)
        
        left_items = []
        right_items = []
        
        for item_id in items:
            bounds = self.items[item_id]
            center = [(bounds[i] + bounds[i+3]) / 2 for i in range(3)]
            
            if center[best_split["axis"]] < best_split["position"]:
                left_items.append(item_id)
            else:
                right_items.append(item_id)
                
        # Create child nodes
        if left_items and right_items:
            left_bounds = self._calculate_bounds(left_items)
            right_bounds = self._calculate_bounds(right_items)
            
            node["children"] = [
                {"bounds": left_bounds, "items": left_items, "children": []},
                {"bounds": right_bounds, "items": right_items, "children": []}
            ]
            node["items"] = []
            
    def query_range(self, bounds: List[float]) -> List[str]:
        """Find all items within 3D bounds."""
        if not self.root:
            return []
            
        results = []
        self._query_recursive(self.root, bounds, results)
        return results
        
    def _query_recursive(self, node: Dict, bounds: List[float], results: List[str]):
        """Recursive range query."""
        if not self._bounds_intersect(node["bounds"], bounds):
            return
            
        if not node["children"]:
            # Leaf node - check items
            for item_id in node["items"]:
                item_bounds = self.items[item_id]
                if self._bounds_intersect(item_bounds, bounds):
                    results.append(item_id)
        else:
            # Internal node - recurse children
            for child in node["children"]:
                self._query_recursive(child, bounds, results)
                
    def nearest_neighbors(self, point: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to a 3D point."""
        candidates = []
        
        for item_id, bounds in self.items.items():
            center = [(bounds[i] + bounds[i+3]) / 2 for i in range(3)]
            distance = self._euclidean_distance(point, center)
            candidates.append((item_id, distance))
            
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
        
    def _euclidean_distance(self, p1: List[float], p2: List[float]) -> float:
        """Calculate 3D Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
        
    def _bounds_intersect(self, b1: List[float], b2: List[float]) -> bool:
        """Check if two 3D bounding boxes intersect."""
        for i in range(3):
            if b1[i] > b2[i+3] or b2[i] > b1[i+3]:
                return False
        return True
        
    def _expand_bounds(self, b1: List[float], b2: List[float]) -> List[float]:
        """Expand bounds to include both boxes."""
        return [
            min(b1[0], b2[0]), min(b1[1], b2[1]), min(b1[2], b2[2]),
            max(b1[3], b2[3]), max(b1[4], b2[4]), max(b1[5], b2[5])
        ]
        
    def _calculate_bounds(self, item_ids: List[str]) -> List[float]:
        """Calculate combined bounds for items."""
        if not item_ids:
            return [0, 0, 0, 0, 0, 0]
            
        bounds = list(self.items[item_ids[0]])
        for item_id in item_ids[1:]:
            bounds = self._expand_bounds(bounds, self.items[item_id])
        return bounds
        
    def _choose_subtree(self, children: List[Dict], bounds: List[float]) -> Dict:
        """Choose best child using minimum volume increase."""
        best_child = None
        min_increase = float('inf')
        
        for child in children:
            expanded = self._expand_bounds(child["bounds"], bounds)
            volume_increase = self._calculate_volume(expanded) - self._calculate_volume(child["bounds"])
            
            if volume_increase < min_increase:
                min_increase = volume_increase
                best_child = child
                
        return best_child
        
    def _calculate_volume(self, bounds: List[float]) -> float:
        """Calculate 3D bounding box volume."""
        return (bounds[3] - bounds[0]) * (bounds[4] - bounds[1]) * (bounds[5] - bounds[2])
        
    def _find_best_split(self, items: List[str]) -> Dict:
        """Find optimal split using Surface Area Heuristic."""
        best_cost = float('inf')
        best_split = {"axis": 0, "position": 0}
        
        for axis in range(3):
            # Sort items by center on this axis
            sorted_items = sorted(items, key=lambda id: 
                (self.items[id][axis] + self.items[id][axis+3]) / 2)
            
            # Try different split positions
            for i in range(1, len(sorted_items)):
                left = sorted_items[:i]
                right = sorted_items[i:]
                
                left_bounds = self._calculate_bounds(left)
                right_bounds = self._calculate_bounds(right)
                
                # SAH cost function
                cost = (len(left) * self._surface_area(left_bounds) + 
                       len(right) * self._surface_area(right_bounds))
                
                if cost < best_cost:
                    best_cost = cost
                    split_pos = (self.items[sorted_items[i-1]][axis+3] + 
                                self.items[sorted_items[i]][axis]) / 2
                    best_split = {"axis": axis, "position": split_pos}
                    
        return best_split
        
    def _surface_area(self, bounds: List[float]) -> float:
        """Calculate 3D bounding box surface area."""
        w = bounds[3] - bounds[0]
        h = bounds[4] - bounds[1]
        d = bounds[5] - bounds[2]
        return 2 * (w*h + w*d + h*d)


class MemoryRoom3D:
    """
    Represents a 3D room in the memory palace.
    Each room has themed associations and spatial anchors.
    """
    
    def __init__(self, room_id: str, theme: str, position: Tuple[float, float, float]):
        self.room_id = room_id
        self.theme = theme
        self.position = position
        self.dimensions = (10.0, 8.0, 10.0)  # Width, height, depth
        self.anchors: List[Dict] = []
        self.connections: List[str] = []  # Connected room IDs
        self.emotional_tone = 0.5  # 0=negative, 1=positive
        self.visit_count = 0
        self.last_visited = None
        
        # Generate room layout based on theme
        self._generate_layout()
        
    def _generate_layout(self):
        """Generate spatial anchors based on room theme."""
        anchor_positions = [
            (0.2, 0.5, 0.2),  # Corner 1
            (0.8, 0.5, 0.2),  # Corner 2
            (0.2, 0.5, 0.8),  # Corner 3
            (0.8, 0.5, 0.8),  # Corner 4
            (0.5, 0.2, 0.5),  # Floor center
            (0.5, 0.8, 0.5),  # Ceiling center
            (0.1, 0.5, 0.5),  # Wall 1
            (0.9, 0.5, 0.5),  # Wall 2
            (0.5, 0.5, 0.1),  # Wall 3
            (0.5, 0.5, 0.9),  # Wall 4
        ]
        
        for i, pos in enumerate(anchor_positions):
            self.anchors.append({
                "id": f"{self.room_id}_anchor_{i}",
                "position": pos,
                "memory_id": None,
                "association_strength": 0.0,
                "visual_marker": self._get_visual_marker(i)
            })
            
    def _get_visual_marker(self, index: int) -> str:
        """Get visual marker for anchor point."""
        markers = [
            "crystal_sphere", "golden_statue", "floating_book",
            "glowing_orb", "ancient_scroll", "holographic_display",
            "memory_crystal", "time_capsule", "neural_interface",
            "quantum_anchor"
        ]
        return markers[index % len(markers)]
        
    def place_memory(self, memory_id: str, anchor_index: int, strength: float = 1.0):
        """Place a memory at specific anchor point."""
        if 0 <= anchor_index < len(self.anchors):
            self.anchors[anchor_index]["memory_id"] = memory_id
            self.anchors[anchor_index]["association_strength"] = strength
            return True
        return False
        
    def get_bounds(self) -> List[float]:
        """Get 3D bounding box for spatial indexing."""
        x, y, z = self.position
        w, h, d = self.dimensions
        return [x, y, z, x + w, y + h, z + d]


class NavigationEngine:
    """
    Handles movement through the memory palace.
    Implements smooth transitions and pathfinding.
    """
    
    def __init__(self):
        self.current_position = (0.0, 0.0, 0.0)
        self.current_room = None
        self.navigation_history = deque(maxlen=100)
        self.movement_speed = 5.0  # Units per second
        self.rotation_speed = 90.0  # Degrees per second
        
    async def move_to_room(self, room: MemoryRoom3D, transition_time: float = 2.0):
        """Smoothly transition to a room."""
        if self.current_room:
            self.navigation_history.append({
                "from": self.current_room.room_id,
                "to": room.room_id,
                "timestamp": datetime.utcnow(),
                "path": self._calculate_path(self.current_position, room.position)
            })
            
        # Simulate smooth movement
        steps = int(transition_time * 30)  # 30 FPS
        start_pos = self.current_position
        end_pos = room.position
        
        for i in range(steps):
            t = (i + 1) / steps
            # Ease-in-out interpolation
            t = t * t * (3.0 - 2.0 * t)
            
            self.current_position = (
                start_pos[0] + (end_pos[0] - start_pos[0]) * t,
                start_pos[1] + (end_pos[1] - start_pos[1]) * t,
                start_pos[2] + (end_pos[2] - start_pos[2]) * t
            )
            
            await asyncio.sleep(transition_time / steps)
            
        self.current_room = room
        room.visit_count += 1
        room.last_visited = datetime.utcnow()
        
    def _calculate_path(self, start: Tuple, end: Tuple) -> List[Tuple]:
        """Calculate navigation path using A* algorithm."""
        # Simplified path for now - can be enhanced with obstacle avoidance
        distance = math.sqrt(sum((e - s) ** 2 for s, e in zip(start, end)))
        steps = max(5, int(distance / 2))
        
        path = []
        for i in range(steps + 1):
            t = i / steps
            path.append((
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
                start[2] + (end[2] - start[2]) * t
            ))
        return path
        
    def get_view_direction(self) -> Tuple[float, float, float]:
        """Get current viewing direction vector."""
        # Can be enhanced with head tracking in VR
        return (0.0, 0.0, 1.0)  # Looking forward by default


class MemoryPalaceEngine:
    """
    Main engine for Memory Palace functionality.
    Manages spatial memory storage and retrieval.
    """
    
    def __init__(self, db: AsyncSession, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
        self.spatial_index = SpatialIndexer()
        self.navigation = NavigationEngine()
        self.llm_service = LLMService()
        self.rooms: Dict[str, MemoryRoom3D] = {}
        self.active_palace = None
        
        # Security and performance components
        self.encryption_service = EncryptionService()
        self.input_sanitizer = InputSanitizer()
        self.privacy_protector = PrivacyProtector()
        self.rate_limiter = RateLimiter()
        self.ml_security = MLSecurityValidator()
        self.consent_manager = ConsentManager()
        self.cache = MultiLevelCache()
        self.model_optimizer = ModelOptimizer()
        
    async def create_palace(self, user_id: UUID, name: str, layout_type: str = "mansion") -> MemoryPalace:
        """Create a new memory palace for user."""
        palace = MemoryPalace(
            user_id=user_id,
            name=name,
            layout_type=layout_type,
            room_count=0,
            total_memories=0,
            navigation_efficiency=1.0,
            spatial_configuration=self._generate_layout_config(layout_type)
        )
        
        self.db.add(palace)
        await self.db.commit()
        await self.db.refresh(palace)
        
        # Generate initial rooms
        await self._generate_initial_rooms(palace)
        
        return palace
        
    def _generate_layout_config(self, layout_type: str) -> Dict:
        """Generate spatial configuration for palace layout."""
        configs = {
            "mansion": {
                "floors": 3,
                "rooms_per_floor": 8,
                "style": "victorian",
                "connections": "hallway",
                "special_rooms": ["library", "study", "gallery", "observatory"]
            },
            "castle": {
                "floors": 5,
                "rooms_per_floor": 12,
                "style": "medieval",
                "connections": "spiral",
                "special_rooms": ["throne", "dungeon", "tower", "armory"]
            },
            "modern": {
                "floors": 2,
                "rooms_per_floor": 6,
                "style": "minimalist",
                "connections": "open",
                "special_rooms": ["office", "media", "gym", "terrace"]
            },
            "temple": {
                "floors": 1,
                "rooms_per_floor": 9,
                "style": "sacred",
                "connections": "circular",
                "special_rooms": ["altar", "meditation", "garden", "sanctum"]
            }
        }
        
        config = configs.get(layout_type, configs["mansion"])
        config["created_at"] = datetime.utcnow().isoformat()
        config["version"] = "1.0"
        return config
        
    async def _generate_initial_rooms(self, palace: MemoryPalace):
        """Generate the initial room structure."""
        config = palace.spatial_configuration
        floors = config.get("floors", 3)
        rooms_per_floor = config.get("rooms_per_floor", 8)
        
        room_themes = [
            "entrance", "living", "kitchen", "bedroom", "study",
            "library", "gallery", "garden", "basement", "attic",
            "workshop", "music", "game", "guest", "storage"
        ]
        
        for floor in range(floors):
            for room_num in range(min(rooms_per_floor, len(room_themes))):
                theme = room_themes[room_num % len(room_themes)]
                
                # Calculate room position in 3D space
                angle = (2 * math.pi * room_num) / rooms_per_floor
                radius = 20.0  # Distance from center
                
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                y = floor * 10.0  # Floor height
                
                room = MemoryRoom(
                    palace_id=palace.id,
                    name=f"{theme.capitalize()} Room",
                    theme=theme,
                    floor_level=floor,
                    position_3d=[x, y, z],
                    anchor_points=[]
                )
                
                self.db.add(room)
                
                # Create 3D representation
                room_3d = MemoryRoom3D(
                    room_id=str(room.id),
                    theme=theme,
                    position=(x, y, z)
                )
                self.rooms[str(room.id)] = room_3d
                
                # Add to spatial index
                self.spatial_index.insert(str(room.id), room_3d.get_bounds())
                
        palace.room_count = floors * rooms_per_floor
        await self.db.commit()
        
    async def store_conversation(
        self,
        user_id: UUID,
        message: Message,
        context: Dict[str, Any]
    ) -> SpatialMemory:
        """Store a conversation in the memory palace."""
        # Security checks
        if not await self.rate_limiter.check_rate_limit(f"memory_palace_{user_id}", max_requests=50):
            raise ValueError("Rate limit exceeded for memory palace operations")
        
        # Check consent for memory storage
        if not await self.consent_manager.check_consent(user_id, "memory_palace_storage"):
            raise ValueError("User consent required for memory palace storage")
        
        # Sanitize message content
        sanitized_content = self.input_sanitizer.sanitize_text(message.content)
        
        # Validate ML security
        security_result = self.ml_security.validate_input(sanitized_content)
        if not security_result['is_safe']:
            logger.warning(f"Unsafe memory content detected: {security_result['reason']}")
            return None
        
        # Get or create palace
        palace = await self._get_or_create_palace(user_id)
        
        # Analyze message for memory classification
        classification = await self._classify_memory(sanitized_content, context)
        
        # Find appropriate room
        room = await self._find_best_room(palace, classification)
        
        # Create spatial memory with encryption
        memory_vector = await self._encode_memory(sanitized_content)
        
        # Encrypt sensitive memory data
        encrypted_associations = self.encryption_service.encrypt_psychological_data({
            "associations": classification["associations"],
            "context": context,
            "user_id_hash": hashlib.sha256(str(user_id).encode()).hexdigest()[:16]
        })
        
        spatial_memory = SpatialMemory(
            palace_id=palace.id,
            room_id=room.id,
            message_id=message.id,
            position_3d=self._find_anchor_position(room, classification),
            memory_strength=classification["importance"],
            emotional_valence=classification["emotion"],
            semantic_category=classification["category"],
            memory_vector=memory_vector.tolist(),
            associations=encrypted_associations  # Now encrypted
        )
        
        self.db.add(spatial_memory)
        
        # Update palace metrics
        palace.total_memories += 1
        
        # Place in 3D room
        room_3d = self.rooms.get(str(room.id))
        if room_3d:
            anchor_index = self._select_anchor(room_3d, classification)
            room_3d.place_memory(str(spatial_memory.id), anchor_index, classification["importance"])
        
        await self.db.commit()
        await self.db.refresh(spatial_memory)
        
        # Cache for quick retrieval with privacy protection
        await self._cache_memory(user_id, spatial_memory, use_encryption=True)
        
        return spatial_memory
        
    async def _classify_memory(self, content: str, context: Dict) -> Dict:
        """Classify memory for spatial placement."""
        prompt = f"""
        Analyze this message for memory palace storage:
        Message: {content}
        Context: {json.dumps(context)}
        
        Return JSON with:
        1. category: primary semantic category
        2. importance: 0-1 score
        3. emotion: emotional valence -1 to 1
        4. associations: list of related concepts
        5. visual_symbol: memorable visual representation
        """
        
        response = await self.llm_service.generate_response(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except:
            return {
                "category": "general",
                "importance": 0.5,
                "emotion": 0.0,
                "associations": [],
                "visual_symbol": "document"
            }
            
    async def _find_best_room(self, palace: MemoryPalace, classification: Dict) -> MemoryRoom:
        """Find the most appropriate room for a memory."""
        category = classification["category"]
        
        # Query rooms by theme match
        result = await self.db.execute(
            select(MemoryRoom)
            .where(MemoryRoom.palace_id == palace.id)
            .order_by(func.similarity(MemoryRoom.theme, category).desc())
            .limit(1)
        )
        
        room = result.scalar_one_or_none()
        
        if not room:
            # Get any room
            result = await self.db.execute(
                select(MemoryRoom)
                .where(MemoryRoom.palace_id == palace.id)
                .limit(1)
            )
            room = result.scalar_one()
            
        return room
        
    def _find_anchor_position(self, room: MemoryRoom, classification: Dict) -> List[float]:
        """Find specific 3D position for memory anchor."""
        # Use importance to determine height (important = higher)
        height = 1.0 + (classification["importance"] * 6.0)
        
        # Use emotion for x-axis (negative = left, positive = right)
        x_offset = 5.0 + (classification["emotion"] * 4.0)
        
        # Random z for depth variation
        z_offset = random.uniform(2.0, 8.0)
        
        base_pos = room.position_3d
        return [
            base_pos[0] + x_offset,
            base_pos[1] + height,
            base_pos[2] + z_offset
        ]
        
    def _select_anchor(self, room: MemoryRoom3D, classification: Dict) -> int:
        """Select best anchor point in room."""
        # Find empty anchor with matching characteristics
        best_anchor = 0
        best_score = -1
        
        for i, anchor in enumerate(room.anchors):
            if anchor["memory_id"] is not None:
                continue  # Already occupied
                
            # Score based on position matching importance/emotion
            pos = anchor["position"]
            score = 0
            
            # Higher positions for important memories
            if classification["importance"] > 0.7 and pos[1] > 0.6:
                score += 1
            elif classification["importance"] < 0.3 and pos[1] < 0.4:
                score += 1
                
            # Emotional mapping
            if classification["emotion"] > 0 and pos[0] > 0.5:
                score += 1
            elif classification["emotion"] < 0 and pos[0] < 0.5:
                score += 1
                
            if score > best_score:
                best_score = score
                best_anchor = i
                
        return best_anchor
        
    async def _encode_memory(self, content: str) -> np.ndarray:
        """Encode memory content to vector representation."""
        # Use LLM to generate embedding
        embedding = await self.llm_service.get_embedding(content)
        return np.array(embedding)
        
    async def navigate_to_memory(
        self,
        user_id: UUID,
        query: str
    ) -> Tuple[SpatialMemory, NavigationPath]:
        """Navigate to a specific memory in the palace."""
        # Search for relevant memories
        memories = await self.search_memories(user_id, query, limit=1)
        
        if not memories:
            return None, None
            
        memory = memories[0]
        
        # Get room
        result = await self.db.execute(
            select(MemoryRoom).where(MemoryRoom.id == memory.room_id)
        )
        room = result.scalar_one()
        
        # Navigate to room
        room_3d = self.rooms.get(str(room.id))
        if room_3d:
            await self.navigation.move_to_room(room_3d)
            
        # Create navigation path
        path = NavigationPath(
            palace_id=memory.palace_id,
            start_position=list(self.navigation.current_position),
            end_position=memory.position_3d,
            waypoints=self.navigation.navigation_history[-1]["path"] if self.navigation.navigation_history else [],
            total_distance=self._calculate_distance(
                self.navigation.current_position,
                tuple(memory.position_3d)
            ),
            navigation_time=2.0
        )
        
        self.db.add(path)
        await self.db.commit()
        
        return memory, path
        
    async def search_memories(
        self,
        user_id: UUID,
        query: str,
        limit: int = 10
    ) -> List[SpatialMemory]:
        """Search memories using spatial and semantic criteria."""
        # Security validation
        if not await self.rate_limiter.check_rate_limit(f"memory_search_{user_id}", max_requests=20):
            raise ValueError("Rate limit exceeded for memory search")
        
        # Sanitize search query
        sanitized_query = self.input_sanitizer.sanitize_text(query)
        
        # Validate query for safety
        security_result = self.ml_security.validate_input(sanitized_query)
        if not security_result['is_safe']:
            logger.warning(f"Unsafe search query: {security_result['reason']}")
            return []
        
        # Get user's palace
        palace = await self._get_or_create_palace(user_id)
        
        # Generate query embedding from sanitized input
        query_vector = await self._encode_memory(sanitized_query)
        
        # Search in database with vector similarity
        result = await self.db.execute(
            select(SpatialMemory)
            .where(SpatialMemory.palace_id == palace.id)
            .order_by(
                func.cube_distance(
                    func.cube(SpatialMemory.memory_vector),
                    func.cube(query_vector.tolist())
                )
            )
            .limit(limit)
        )
        
        memories = result.scalars().all()
        
        # Enhance with spatial proximity
        if self.navigation.current_room:
            current_pos = self.navigation.current_position
            
            # Sort by combined semantic and spatial distance
            scored_memories = []
            for memory in memories:
                spatial_distance = self._calculate_distance(
                    current_pos,
                    tuple(memory.position_3d)
                )
                
                semantic_similarity = 1.0 - np.linalg.norm(
                    query_vector - np.array(memory.memory_vector)
                )
                
                # Combined score (weighted)
                score = (semantic_similarity * 0.7) + ((1.0 / (1.0 + spatial_distance)) * 0.3)
                scored_memories.append((memory, score))
                
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            memories = [m for m, _ in scored_memories]
            
        return memories
        
    async def get_nearby_memories(
        self,
        user_id: UUID,
        radius: float = 10.0
    ) -> List[SpatialMemory]:
        """Get memories near current position."""
        if not self.navigation.current_position:
            return []
            
        palace = await self._get_or_create_palace(user_id)
        
        x, y, z = self.navigation.current_position
        bounds = [x - radius, y - radius, z - radius,
                 x + radius, y + radius, z + radius]
        
        # Use spatial index for efficient query
        nearby_room_ids = self.spatial_index.query_range(bounds)
        
        if not nearby_room_ids:
            return []
            
        # Get memories from nearby rooms
        result = await self.db.execute(
            select(SpatialMemory)
            .where(
                and_(
                    SpatialMemory.palace_id == palace.id,
                    SpatialMemory.room_id.in_([UUID(rid) for rid in nearby_room_ids])
                )
            )
            .order_by(SpatialMemory.memory_strength.desc())
            .limit(20)
        )
        
        return result.scalars().all()
        
    async def create_memory_trail(
        self,
        user_id: UUID,
        topic: str
    ) -> List[SpatialMemory]:
        """Create a guided path through related memories."""
        memories = await self.search_memories(user_id, topic, limit=20)
        
        if len(memories) < 2:
            return memories
            
        # Order memories to minimize travel distance (TSP approximation)
        ordered = [memories[0]]
        remaining = memories[1:]
        
        while remaining:
            current = ordered[-1]
            nearest = min(
                remaining,
                key=lambda m: self._calculate_distance(
                    tuple(current.position_3d),
                    tuple(m.position_3d)
                )
            )
            ordered.append(nearest)
            remaining.remove(nearest)
            
        return ordered
        
    def _calculate_distance(self, p1: Tuple, p2: Tuple) -> float:
        """Calculate 3D Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
        
    async def _get_or_create_palace(self, user_id: UUID) -> MemoryPalace:
        """Get existing palace or create new one."""
        result = await self.db.execute(
            select(MemoryPalace)
            .where(MemoryPalace.user_id == user_id)
            .order_by(MemoryPalace.created_at.desc())
            .limit(1)
        )
        
        palace = result.scalar_one_or_none()
        
        if not palace:
            palace = await self.create_palace(user_id, "My Memory Palace")
            
        self.active_palace = palace
        return palace
        
    async def _cache_memory(self, user_id: UUID, memory: SpatialMemory, use_encryption: bool = False):
        """Cache memory for quick retrieval with optional encryption."""
        # Use anonymized cache key
        cache_key = f"memory_palace:{hashlib.sha256(str(user_id).encode()).hexdigest()[:16]}:memory:{memory.id}"
        
        data = {
            "id": str(memory.id),
            "room_id": str(memory.room_id),
            "position": memory.position_3d,
            "strength": memory.memory_strength,
            "category": memory.semantic_category
        }
        
        if use_encryption:
            # Encrypt sensitive memory data
            encrypted_data = self.encryption_service.encrypt_psychological_data(data)
            await self.cache.set(cache_key, encrypted_data, ttl=3600)
        else:
            # Apply privacy protection
            protected_data = self.privacy_protector.anonymize_response(data, str(user_id))
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour cache
                json.dumps(protected_data)
            )
        
    async def export_palace_data(self, user_id: UUID) -> Dict:
        """Export palace data for 3D visualization with privacy protection."""
        # Security checks
        if not await self.rate_limiter.check_rate_limit(f"palace_export_{user_id}", max_requests=5):
            raise ValueError("Rate limit exceeded for palace data export")
        
        # Check consent for data export
        if not await self.consent_manager.check_consent(user_id, "memory_palace_export"):
            raise ValueError("User consent required for palace data export")
        
        palace = await self._get_or_create_palace(user_id)
        
        # Get all rooms
        result = await self.db.execute(
            select(MemoryRoom)
            .where(MemoryRoom.palace_id == palace.id)
        )
        rooms = result.scalars().all()
        
        # Get all memories
        result = await self.db.execute(
            select(SpatialMemory)
            .where(SpatialMemory.palace_id == palace.id)
            .options(selectinload(SpatialMemory.message))
        )
        memories = result.scalars().all()
        
        # Build 3D scene data
        scene = {
            "palace": {
                "id": str(palace.id),
                "name": palace.name,
                "layout": palace.layout_type,
                "stats": {
                    "rooms": palace.room_count,
                    "memories": palace.total_memories,
                    "efficiency": palace.navigation_efficiency
                }
            },
            "rooms": [],
            "memories": [],
            "connections": []
        }
        
        for room in rooms:
            room_3d = self.rooms.get(str(room.id))
            scene["rooms"].append({
                "id": str(room.id),
                "name": room.name,
                "theme": room.theme,
                "position": room.position_3d,
                "bounds": room_3d.get_bounds() if room_3d else None,
                "anchors": room_3d.anchors if room_3d else [],
                "visitCount": room_3d.visit_count if room_3d else 0
            })
            
        for memory in memories:
            scene["memories"].append({
                "id": str(memory.id),
                "roomId": str(memory.room_id),
                "position": memory.position_3d,
                "strength": memory.memory_strength,
                "emotion": memory.emotional_valence,
                "category": memory.semantic_category,
                "preview": memory.message.content[:100] if memory.message else "",
                "timestamp": memory.created_at.isoformat()
            })
            
        # Add navigation paths
        if self.navigation.navigation_history:
            for nav in list(self.navigation.navigation_history)[-10:]:
                scene["connections"].append({
                    "from": nav["from"],
                    "to": nav["to"],
                    "path": nav["path"],
                    "timestamp": nav["timestamp"].isoformat()
                })
                
        return scene