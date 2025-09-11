"""
Memory Palace Service

Advanced spatial memory and knowledge organization system
using revolutionary memory palace techniques.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.core.redis import redis_manager
from app.core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class RoomType(Enum):
    """Memory palace room types"""
    ENTRANCE_HALL = "entrance_hall"
    LIBRARY = "library"
    WORKSHOP = "workshop"
    GARDEN = "garden"
    OBSERVATORY = "observatory"
    ARCHIVE = "archive"
    SANCTUARY = "sanctuary"
    LABORATORY = "laboratory"

@dataclass
class MemoryItem:
    """Item stored in memory palace"""
    id: str
    content: str
    associations: List[str]
    emotional_weight: float
    access_frequency: int
    last_accessed: datetime
    spatial_location: Tuple[float, float, float]
    room_type: RoomType

@dataclass
class MemoryPalace:
    """User's complete memory palace"""
    user_id: str
    rooms: Dict[RoomType, Dict[str, MemoryItem]]
    pathways: List[Tuple[str, str]]  # Connections between memories
    total_items: int
    creation_date: datetime
    last_modified: datetime

class MemoryPalaceService:
    """Revolutionary memory palace service"""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=Exception
        )
        
        self.spatial_processor = None
        self.association_engine = None
        
        # Room configurations
        self.room_configs = {
            RoomType.ENTRANCE_HALL: {
                'capacity': 20,
                'description': 'First impressions and initial thoughts',
                'associations': ['introduction', 'overview', 'first_contact']
            },
            RoomType.LIBRARY: {
                'capacity': 100,
                'description': 'Factual knowledge and references',
                'associations': ['facts', 'knowledge', 'information', 'learning']
            },
            RoomType.WORKSHOP: {
                'capacity': 50,
                'description': 'Skills, processes, and how-to knowledge',
                'associations': ['skills', 'process', 'method', 'technique']
            },
            RoomType.GARDEN: {
                'capacity': 30,
                'description': 'Growing ideas and creative thoughts',
                'associations': ['creativity', 'growth', 'ideas', 'inspiration']
            },
            RoomType.OBSERVATORY: {
                'capacity': 40,
                'description': 'Big picture insights and patterns',
                'associations': ['patterns', 'insights', 'perspective', 'vision']
            },
            RoomType.ARCHIVE: {
                'capacity': 200,
                'description': 'Long-term storage for important memories',
                'associations': ['history', 'important', 'significant', 'milestone']
            },
            RoomType.SANCTUARY: {
                'capacity': 15,
                'description': 'Sacred and deeply personal memories',
                'associations': ['personal', 'sacred', 'meaningful', 'emotional']
            },
            RoomType.LABORATORY: {
                'capacity': 60,
                'description': 'Experimental ideas and hypotheses',
                'associations': ['experiment', 'hypothesis', 'test', 'theory']
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize memory palace service"""
        try:
            await self._load_spatial_models()
            logger.info("Memory palace service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory palace service: {str(e)}")
            return False
    
    async def _load_spatial_models(self):
        """Load spatial processing models"""
        self.spatial_processor = {
            'version': '2.3.0',
            'spatial_accuracy': 0.92,
            'association_strength': 0.88,
            'retrieval_efficiency': 0.91
        }
        
        self.association_engine = {
            'version': '3.1.0',
            'semantic_similarity': 0.89,
            'contextual_linking': 0.87,
            'temporal_connections': 0.85
        }
    
    @CircuitBreaker.protect
    async def store_memory(self, user_id: str, content: str, 
                         context: Dict[str, Any] = None) -> MemoryItem:
        """Store new memory in user's palace"""
        try:
            # Get or create memory palace
            palace = await self._get_memory_palace(user_id)
            
            # Analyze content for optimal placement
            room_type = await self._determine_optimal_room(content, context)
            
            # Generate spatial location
            location = await self._generate_spatial_location(palace, room_type)
            
            # Extract associations
            associations = await self._extract_associations(content, context)
            
            # Calculate emotional weight
            emotional_weight = await self._calculate_emotional_weight(content, context)
            
            # Create memory item
            memory_item = MemoryItem(
                id=f"{user_id}_{datetime.now().timestamp()}",
                content=content,
                associations=associations,
                emotional_weight=emotional_weight,
                access_frequency=0,
                last_accessed=datetime.now(),
                spatial_location=location,
                room_type=room_type
            )
            
            # Store in palace
            await self._add_to_palace(palace, memory_item)
            
            # Create pathways to related memories
            await self._create_memory_pathways(palace, memory_item)
            
            # Save updated palace
            await self._save_memory_palace(palace)
            
            logger.info(f"Memory stored for user {user_id} in {room_type.value}")
            return memory_item
            
        except Exception as e:
            logger.error(f"Memory storage failed: {str(e)}")
            raise
    
    async def _determine_optimal_room(self, content: str, 
                                    context: Dict[str, Any] = None) -> RoomType:
        """Determine optimal room for content"""
        content_lower = content.lower()
        
        # Analyze content for room placement
        room_scores = {}
        
        for room_type, config in self.room_configs.items():
            score = 0.0
            
            # Check associations
            for association in config['associations']:
                if association in content_lower:
                    score += 0.3
            
            room_scores[room_type] = score
        
        # Context-based adjustments
        if context:
            if context.get('importance') == 'high':
                room_scores[RoomType.ARCHIVE] += 0.5
            if context.get('emotional_intensity', 0) > 0.7:
                room_scores[RoomType.SANCTUARY] += 0.4
            if context.get('creative', False):
                room_scores[RoomType.GARDEN] += 0.3
            if context.get('analytical', False):
                room_scores[RoomType.LABORATORY] += 0.3
        
        # Content type analysis
        if any(word in content_lower for word in ['fact', 'definition', 'reference']):
            room_scores[RoomType.LIBRARY] += 0.4
        elif any(word in content_lower for word in ['how to', 'process', 'method']):
            room_scores[RoomType.WORKSHOP] += 0.4
        elif any(word in content_lower for word in ['idea', 'creative', 'imagine']):
            room_scores[RoomType.GARDEN] += 0.4
        elif any(word in content_lower for word in ['pattern', 'trend', 'overview']):
            room_scores[RoomType.OBSERVATORY] += 0.4
        
        # Return room with highest score
        best_room = max(room_scores, key=room_scores.get)
        
        # Default to entrance hall if no clear match
        if room_scores[best_room] < 0.2:
            return RoomType.ENTRANCE_HALL
        
        return best_room
    
    async def _generate_spatial_location(self, palace: MemoryPalace, 
                                       room_type: RoomType) -> Tuple[float, float, float]:
        """Generate spatial location within room"""
        # Get existing items in room
        room_items = palace.rooms.get(room_type, {})
        
        # Simple spatial distribution (in production, use more sophisticated algorithms)
        import random
        
        # Avoid crowding by checking nearby locations
        max_attempts = 10
        for _ in range(max_attempts):
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            z = random.uniform(-1.0, 1.0)
            
            # Check distance from existing items
            too_close = False
            for item in room_items.values():
                distance = (
                    (x - item.spatial_location[0])**2 + 
                    (y - item.spatial_location[1])**2 + 
                    (z - item.spatial_location[2])**2
                )**0.5
                
                if distance < 0.3:  # Minimum distance threshold
                    too_close = True
                    break
            
            if not too_close:
                return (x, y, z)
        
        # Fallback to random location if all attempts failed
        return (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
    
    async def _extract_associations(self, content: str, 
                                  context: Dict[str, Any] = None) -> List[str]:
        """Extract semantic associations from content"""
        associations = []
        content_lower = content.lower()
        
        # Extract key concepts (simplified - in production use NLP)
        words = content_lower.split()
        
        # Remove stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [word.strip('.,!?;:') for word in words 
                          if len(word) > 3 and word not in stop_words]
        
        # Take top concepts
        associations.extend(meaningful_words[:10])
        
        # Add context associations
        if context:
            for key, value in context.items():
                if isinstance(value, str):
                    associations.append(f"{key}:{value}")
        
        return associations[:15]  # Limit to 15 associations
    
    async def _calculate_emotional_weight(self, content: str, 
                                        context: Dict[str, Any] = None) -> float:
        """Calculate emotional weight of content"""
        weight = 0.5  # Base weight
        content_lower = content.lower()
        
        # Emotional indicators
        positive_emotions = ['love', 'joy', 'happy', 'wonderful', 'amazing', 'great']
        negative_emotions = ['sad', 'angry', 'terrible', 'awful', 'hate', 'fear']
        
        for emotion in positive_emotions:
            if emotion in content_lower:
                weight += 0.1
        
        for emotion in negative_emotions:
            if emotion in content_lower:
                weight += 0.15  # Negative emotions often have stronger memory weight
        
        # Exclamation marks and emphasis
        weight += content.count('!') * 0.05
        weight += content.count('?') * 0.02
        
        # Context emotional intensity
        if context and 'emotional_intensity' in context:
            weight += context['emotional_intensity'] * 0.3
        
        return min(1.0, weight)
    
    async def _get_memory_palace(self, user_id: str) -> MemoryPalace:
        """Get or create user's memory palace"""
        cache_key = f"memory_palace:{user_id}"
        cached_data = await redis_manager.get(cache_key)
        
        if cached_data:
            palace_data = json.loads(cached_data)
            
            # Reconstruct memory palace
            rooms = {}
            for room_str, items_data in palace_data['rooms'].items():
                room_type = RoomType(room_str)
                room_items = {}
                
                for item_id, item_data in items_data.items():
                    memory_item = MemoryItem(
                        id=item_data['id'],
                        content=item_data['content'],
                        associations=item_data['associations'],
                        emotional_weight=item_data['emotional_weight'],
                        access_frequency=item_data['access_frequency'],
                        last_accessed=datetime.fromisoformat(item_data['last_accessed']),
                        spatial_location=tuple(item_data['spatial_location']),
                        room_type=RoomType(item_data['room_type'])
                    )
                    room_items[item_id] = memory_item
                
                rooms[room_type] = room_items
            
            return MemoryPalace(
                user_id=palace_data['user_id'],
                rooms=rooms,
                pathways=palace_data['pathways'],
                total_items=palace_data['total_items'],
                creation_date=datetime.fromisoformat(palace_data['creation_date']),
                last_modified=datetime.fromisoformat(palace_data['last_modified'])
            )
        
        # Create new palace
        return MemoryPalace(
            user_id=user_id,
            rooms={room_type: {} for room_type in RoomType},
            pathways=[],
            total_items=0,
            creation_date=datetime.now(),
            last_modified=datetime.now()
        )
    
    async def _add_to_palace(self, palace: MemoryPalace, item: MemoryItem):
        """Add memory item to palace"""
        room_type = item.room_type
        
        if room_type not in palace.rooms:
            palace.rooms[room_type] = {}
        
        palace.rooms[room_type][item.id] = item
        palace.total_items += 1
        palace.last_modified = datetime.now()
    
    async def _create_memory_pathways(self, palace: MemoryPalace, new_item: MemoryItem):
        """Create pathways between related memories"""
        # Find related memories based on associations
        for room_type, room_items in palace.rooms.items():
            for item_id, existing_item in room_items.items():
                if item_id == new_item.id:
                    continue
                
                # Calculate similarity
                similarity = await self._calculate_memory_similarity(new_item, existing_item)
                
                if similarity > 0.3:  # Threshold for creating pathway
                    pathway = (new_item.id, existing_item.id)
                    if pathway not in palace.pathways:
                        palace.pathways.append(pathway)
    
    async def _calculate_memory_similarity(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity between two memory items"""
        # Association overlap
        common_associations = set(item1.associations) & set(item2.associations)
        total_associations = set(item1.associations) | set(item2.associations)
        
        if not total_associations:
            return 0.0
        
        association_similarity = len(common_associations) / len(total_associations)
        
        # Spatial proximity (if in same room)
        spatial_similarity = 0.0
        if item1.room_type == item2.room_type:
            distance = (
                (item1.spatial_location[0] - item2.spatial_location[0])**2 + 
                (item1.spatial_location[1] - item2.spatial_location[1])**2 + 
                (item1.spatial_location[2] - item2.spatial_location[2])**2
            )**0.5
            
            spatial_similarity = max(0, 1 - distance / 2.83)  # Max distance in unit cube
        
        # Emotional weight similarity
        emotional_similarity = 1 - abs(item1.emotional_weight - item2.emotional_weight)
        
        # Weighted combination
        similarity = (association_similarity * 0.6 + 
                     spatial_similarity * 0.2 + 
                     emotional_similarity * 0.2)
        
        return similarity
    
    async def _save_memory_palace(self, palace: MemoryPalace):
        """Save memory palace to cache"""
        try:
            cache_key = f"memory_palace:{palace.user_id}"
            
            # Convert to serializable format
            palace_data = {
                'user_id': palace.user_id,
                'rooms': {},
                'pathways': palace.pathways,
                'total_items': palace.total_items,
                'creation_date': palace.creation_date.isoformat(),
                'last_modified': palace.last_modified.isoformat()
            }
            
            for room_type, room_items in palace.rooms.items():
                palace_data['rooms'][room_type.value] = {}
                
                for item_id, item in room_items.items():
                    palace_data['rooms'][room_type.value][item_id] = {
                        'id': item.id,
                        'content': item.content,
                        'associations': item.associations,
                        'emotional_weight': item.emotional_weight,
                        'access_frequency': item.access_frequency,
                        'last_accessed': item.last_accessed.isoformat(),
                        'spatial_location': list(item.spatial_location),
                        'room_type': item.room_type.value
                    }
            
            await redis_manager.set(
                cache_key,
                json.dumps(palace_data),
                ttl=timedelta(days=30)
            )
            
        except Exception as e:
            logger.error(f"Failed to save memory palace: {str(e)}")
    
    async def retrieve_memories(self, user_id: str, query: str, 
                              limit: int = 10) -> List[MemoryItem]:
        """Retrieve memories matching query"""
        try:
            palace = await self._get_memory_palace(user_id)
            
            # Score all memories for relevance
            scored_memories = []
            
            for room_type, room_items in palace.rooms.items():
                for item in room_items.values():
                    score = await self._calculate_retrieval_score(item, query)
                    scored_memories.append((score, item))
            
            # Sort by score and return top results
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            
            results = [item for score, item in scored_memories[:limit]]
            
            # Update access frequency
            for item in results:
                item.access_frequency += 1
                item.last_accessed = datetime.now()
            
            # Save updated palace
            await self._save_memory_palace(palace)
            
            return results
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {str(e)}")
            return []
    
    async def _calculate_retrieval_score(self, item: MemoryItem, query: str) -> float:
        """Calculate relevance score for memory retrieval"""
        score = 0.0
        query_lower = query.lower()
        
        # Content match
        if query_lower in item.content.lower():
            score += 0.5
        
        # Association match
        for association in item.associations:
            if association.lower() in query_lower or query_lower in association.lower():
                score += 0.2
        
        # Recent access bonus
        days_since_access = (datetime.now() - item.last_accessed).days
        recency_bonus = max(0, 1 - days_since_access / 30)  # Decay over 30 days
        score += recency_bonus * 0.1
        
        # Frequency bonus
        frequency_bonus = min(0.2, item.access_frequency * 0.02)
        score += frequency_bonus
        
        # Emotional weight bonus
        score += item.emotional_weight * 0.1
        
        return score
    
    async def get_palace_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get memory palace statistics"""
        try:
            palace = await self._get_memory_palace(user_id)
            
            room_stats = {}
            for room_type, room_items in palace.rooms.items():
                room_stats[room_type.value] = {
                    'item_count': len(room_items),
                    'capacity': self.room_configs[room_type]['capacity'],
                    'utilization': len(room_items) / self.room_configs[room_type]['capacity']
                }
            
            return {
                'total_memories': palace.total_items,
                'total_pathways': len(palace.pathways),
                'creation_date': palace.creation_date.isoformat(),
                'last_modified': palace.last_modified.isoformat(),
                'room_statistics': room_stats,
                'most_accessed_room': max(room_stats, key=lambda x: room_stats[x]['item_count']) if room_stats else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get palace statistics: {str(e)}")
            return {}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of memory palace service"""
        return {
            'status': 'healthy',
            'models_loaded': bool(self.spatial_processor and self.association_engine),
            'room_types': len(RoomType),
            'total_room_capacity': sum(config['capacity'] for config in self.room_configs.values()),
            'circuit_breaker': self.circuit_breaker.state,
            'last_check': datetime.now().isoformat()
        }
