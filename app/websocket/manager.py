"""
WebSocket Connection Manager

Manages all WebSocket connections for real-time features including:
- Consciousness mirroring real-time updates
- Emotional intelligence live feedback
- Quantum consciousness state broadcasting
- Digital telepathy network communication
- Neural dream streaming
- Meta-reality session updates
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from app.config import settings

logger = logging.getLogger(__name__)

class WebSocketConnection:
    """Represents a single WebSocket connection"""
    
    def __init__(self, websocket: WebSocket, connection_id: str, user_id: Optional[int] = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.subscribed_topics: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
        
    async def send_message(self, message_type: str, data: Any):
        """Send a message to this connection"""
        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "connection_id": self.connection_id
            }
            await self.websocket.send_text(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Failed to send message to connection {self.connection_id}: {str(e)}")
            raise
    
    async def ping(self):
        """Send ping to keep connection alive"""
        try:
            await self.send_message("ping", {"timestamp": datetime.utcnow().isoformat()})
            self.last_ping = datetime.utcnow()
            
        except Exception as e:
            logger.debug(f"Ping failed for connection {self.connection_id}: {str(e)}")
            raise
    
    def is_stale(self, timeout_minutes: int = 5) -> bool:
        """Check if connection is stale (no ping response)"""
        return datetime.utcnow() - self.last_ping > timedelta(minutes=timeout_minutes)
    
    def subscribe_to_topic(self, topic: str):
        """Subscribe to a specific topic"""
        self.subscribed_topics.add(topic)
        logger.debug(f"Connection {self.connection_id} subscribed to topic: {topic}")
    
    def unsubscribe_from_topic(self, topic: str):
        """Unsubscribe from a specific topic"""
        self.subscribed_topics.discard(topic)
        logger.debug(f"Connection {self.connection_id} unsubscribed from topic: {topic}")

class WebSocketManager:
    """Manages all WebSocket connections and real-time communications"""
    
    def __init__(self):
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[int, Set[str]] = {}  # user_id -> connection_ids
        self.topic_subscribers: Dict[str, Set[str]] = {}  # topic -> connection_ids
        
        # AI session tracking
        self.ai_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.message_count = 0
        self.connection_count = 0
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the WebSocket manager"""
        if self.initialized:
            return
            
        logger.info("Initializing WebSocket manager...")
        
        # Start periodic tasks
        asyncio.create_task(self._periodic_ping_task())
        asyncio.create_task(self._periodic_cleanup_task())
        
        self.initialized = True
        logger.info("WebSocket manager initialized")
    
    async def connect(self, websocket: WebSocket, user_id: Optional[int] = None) -> str:
        """Accept a new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        
        await websocket.accept()
        
        connection = WebSocketConnection(websocket, connection_id, user_id)
        self.connections[connection_id] = connection
        
        # Track user connections
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
        
        self.connection_count = len(self.connections)
        
        logger.info(f"WebSocket connection established: {connection_id} (user: {user_id})")
        
        # Send welcome message
        await connection.send_message("welcome", {
            "connection_id": connection_id,
            "server_time": datetime.utcnow().isoformat(),
            "features_available": [
                "consciousness_mirroring",
                "emotional_intelligence", 
                "quantum_consciousness",
                "digital_telepathy",
                "neural_dreams",
                "meta_reality",
                "transcendence_protocol"
            ]
        })
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection"""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        
        # Remove from user connections
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        # Remove from topic subscriptions
        for topic in connection.subscribed_topics:
            if topic in self.topic_subscribers:
                self.topic_subscribers[topic].discard(connection_id)
                if not self.topic_subscribers[topic]:
                    del self.topic_subscribers[topic]
        
        # Remove connection
        del self.connections[connection_id]
        self.connection_count = len(self.connections)
        
        logger.info(f"WebSocket connection disconnected: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message_type: str, data: Any) -> bool:
        """Send message to a specific connection"""
        if connection_id not in self.connections:
            return False
            
        try:
            connection = self.connections[connection_id]
            await connection.send_message(message_type, data)
            self.message_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {str(e)}")
            await self.disconnect(connection_id)
            return False
    
    async def send_to_user(self, user_id: int, message_type: str, data: Any) -> int:
        """Send message to all connections for a specific user"""
        if user_id not in self.user_connections:
            return 0
            
        sent_count = 0
        for connection_id in list(self.user_connections[user_id]):
            if await self.send_to_connection(connection_id, message_type, data):
                sent_count += 1
                
        return sent_count
    
    async def broadcast_to_topic(self, topic: str, message_type: str, data: Any) -> int:
        """Broadcast message to all subscribers of a topic"""
        if topic not in self.topic_subscribers:
            return 0
            
        sent_count = 0
        for connection_id in list(self.topic_subscribers[topic]):
            if await self.send_to_connection(connection_id, message_type, data):
                sent_count += 1
                
        return sent_count
    
    async def broadcast_to_all(self, message_type: str, data: Any) -> int:
        """Broadcast message to all connected clients"""
        sent_count = 0
        for connection_id in list(self.connections.keys()):
            if await self.send_to_connection(connection_id, message_type, data):
                sent_count += 1
                
        return sent_count
    
    async def subscribe_to_topic(self, connection_id: str, topic: str) -> bool:
        """Subscribe a connection to a topic"""
        if connection_id not in self.connections:
            return False
            
        connection = self.connections[connection_id]
        connection.subscribe_to_topic(topic)
        
        # Add to topic subscribers
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(connection_id)
        
        # Send subscription confirmation
        await connection.send_message("subscription_confirmed", {
            "topic": topic,
            "subscriber_count": len(self.topic_subscribers[topic])
        })
        
        return True
    
    async def unsubscribe_from_topic(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe a connection from a topic"""
        if connection_id not in self.connections:
            return False
            
        connection = self.connections[connection_id]
        connection.unsubscribe_from_topic(topic)
        
        # Remove from topic subscribers
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(connection_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
        
        # Send unsubscription confirmation
        await connection.send_message("unsubscription_confirmed", {
            "topic": topic
        })
        
        return True
    
    # AI-specific methods
    
    async def start_consciousness_session(self, connection_id: str, session_config: Dict[str, Any]) -> str:
        """Start a consciousness mirroring session"""
        session_id = str(uuid.uuid4())
        
        self.ai_sessions[session_id] = {
            "type": "consciousness",
            "connection_id": connection_id,
            "config": session_config,
            "started_at": datetime.utcnow(),
            "status": "active"
        }
        
        # Subscribe to consciousness updates
        await self.subscribe_to_topic(connection_id, f"consciousness_{session_id}")
        
        # Send session started confirmation
        await self.send_to_connection(connection_id, "consciousness_session_started", {
            "session_id": session_id,
            "config": session_config
        })
        
        logger.info(f"Consciousness session started: {session_id}")
        return session_id
    
    async def broadcast_consciousness_update(self, session_id: str, update_data: Dict[str, Any]):
        """Broadcast consciousness state update"""
        topic = f"consciousness_{session_id}"
        await self.broadcast_to_topic(topic, "consciousness_update", {
            "session_id": session_id,
            "update": update_data
        })
    
    async def start_emotional_intelligence_session(self, connection_id: str, session_config: Dict[str, Any]) -> str:
        """Start an emotional intelligence session"""
        session_id = str(uuid.uuid4())
        
        self.ai_sessions[session_id] = {
            "type": "emotional_intelligence",
            "connection_id": connection_id,
            "config": session_config,
            "started_at": datetime.utcnow(),
            "status": "active"
        }
        
        # Subscribe to emotional updates
        await self.subscribe_to_topic(connection_id, f"emotional_{session_id}")
        
        await self.send_to_connection(connection_id, "emotional_session_started", {
            "session_id": session_id,
            "config": session_config
        })
        
        logger.info(f"Emotional intelligence session started: {session_id}")
        return session_id
    
    async def broadcast_emotional_update(self, session_id: str, emotional_data: Dict[str, Any]):
        """Broadcast emotional intelligence update"""
        topic = f"emotional_{session_id}"
        await self.broadcast_to_topic(topic, "emotional_update", {
            "session_id": session_id,
            "emotional_state": emotional_data
        })
    
    async def start_quantum_consciousness_session(self, connection_id: str, session_config: Dict[str, Any]) -> str:
        """Start a quantum consciousness session"""
        session_id = str(uuid.uuid4())
        
        self.ai_sessions[session_id] = {
            "type": "quantum_consciousness",
            "connection_id": connection_id,
            "config": session_config,
            "started_at": datetime.utcnow(),
            "status": "active"
        }
        
        # Subscribe to quantum updates
        await self.subscribe_to_topic(connection_id, f"quantum_{session_id}")
        
        await self.send_to_connection(connection_id, "quantum_session_started", {
            "session_id": session_id,
            "config": session_config
        })
        
        logger.info(f"Quantum consciousness session started: {session_id}")
        return session_id
    
    async def broadcast_quantum_update(self, session_id: str, quantum_data: Dict[str, Any]):
        """Broadcast quantum consciousness update"""
        topic = f"quantum_{session_id}"
        await self.broadcast_to_topic(topic, "quantum_update", {
            "session_id": session_id,
            "quantum_state": quantum_data
        })
    
    async def start_digital_telepathy_session(self, connection_id: str, session_config: Dict[str, Any]) -> str:
        """Start a digital telepathy session"""
        session_id = str(uuid.uuid4())
        
        self.ai_sessions[session_id] = {
            "type": "digital_telepathy",
            "connection_id": connection_id,
            "config": session_config,
            "started_at": datetime.utcnow(),
            "status": "active",
            "participants": []
        }
        
        # Subscribe to telepathy updates
        await self.subscribe_to_topic(connection_id, f"telepathy_{session_id}")
        
        await self.send_to_connection(connection_id, "telepathy_session_started", {
            "session_id": session_id,
            "config": session_config
        })
        
        logger.info(f"Digital telepathy session started: {session_id}")
        return session_id
    
    async def broadcast_telepathy_transmission(self, session_id: str, transmission_data: Dict[str, Any]):
        """Broadcast telepathy transmission"""
        topic = f"telepathy_{session_id}"
        await self.broadcast_to_topic(topic, "telepathy_transmission", {
            "session_id": session_id,
            "transmission": transmission_data
        })
    
    async def stream_neural_dream(self, connection_id: str, dream_data: Dict[str, Any]):
        """Stream neural dream content"""
        await self.send_to_connection(connection_id, "neural_dream_stream", {
            "dream_id": dream_data.get("dream_id"),
            "content": dream_data.get("content"),
            "imagery": dream_data.get("imagery"),
            "narrative": dream_data.get("narrative")
        })
    
    async def end_ai_session(self, session_id: str):
        """End an AI session"""
        if session_id in self.ai_sessions:
            session = self.ai_sessions[session_id]
            session["status"] = "ended"
            session["ended_at"] = datetime.utcnow()
            
            # Notify connection
            connection_id = session["connection_id"]
            await self.send_to_connection(connection_id, "ai_session_ended", {
                "session_id": session_id,
                "session_type": session["type"]
            })
            
            # Clean up topic subscriptions
            topic_prefix = session["type"]
            topic = f"{topic_prefix}_{session_id}"
            await self.unsubscribe_from_topic(connection_id, topic)
            
            del self.ai_sessions[session_id]
            logger.info(f"AI session ended: {session_id}")
    
    # Maintenance methods
    
    async def cleanup_inactive_connections(self):
        """Remove inactive/stale connections"""
        stale_connections = []
        
        for connection_id, connection in self.connections.items():
            if connection.is_stale():
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            logger.info(f"Cleaning up stale connection: {connection_id}")
            await self.disconnect(connection_id)
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale connections")
    
    async def _periodic_ping_task(self):
        """Periodically ping all connections"""
        while True:
            try:
                if self.connections:
                    logger.debug(f"Pinging {len(self.connections)} connections")
                    
                    for connection_id, connection in list(self.connections.items()):
                        try:
                            await connection.ping()
                        except Exception:
                            # Connection failed, will be cleaned up later
                            pass
                
                await asyncio.sleep(30)  # Ping every 30 seconds
                
            except Exception as e:
                logger.error(f"Ping task error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _periodic_cleanup_task(self):
        """Periodically clean up inactive connections"""
        while True:
            try:
                await self.cleanup_inactive_connections()
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")
                await asyncio.sleep(120)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            "total_connections": len(self.connections),
            "total_users": len(self.user_connections),
            "total_topics": len(self.topic_subscribers),
            "active_ai_sessions": len(self.ai_sessions),
            "messages_sent": self.message_count,
            "topics": list(self.topic_subscribers.keys()),
            "ai_session_types": [session["type"] for session in self.ai_sessions.values()],
            "connection_health": {
                connection_id: {
                    "user_id": conn.user_id,
                    "connected_at": conn.connected_at.isoformat(),
                    "last_ping": conn.last_ping.isoformat(),
                    "subscribed_topics": list(conn.subscribed_topics),
                    "is_stale": conn.is_stale()
                }
                for connection_id, conn in self.connections.items()
            }
        }

# Global WebSocket manager instance
websocket_manager = WebSocketManager()