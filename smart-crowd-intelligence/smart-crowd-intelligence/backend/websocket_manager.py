"""
WebSocket Manager
Handles real-time WebSocket connections for live data streaming
"""

from fastapi import WebSocket
import json
import logging
from typing import List, Dict, Set
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.stream_subscribers: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "connected_at": asyncio.get_event_loop().time(),
            "subscribed_streams": set()
        }
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message(websocket, {
            "type": "connection_established",
            "message": "Connected to Smart Crowd Intelligence System",
            "connection_id": id(websocket)
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            # Remove from stream subscriptions
            if websocket in self.connection_metadata:
                subscribed_streams = self.connection_metadata[websocket]["subscribed_streams"]
                for stream_id in subscribed_streams:
                    self.stream_subscribers[stream_id].discard(websocket)
                
                del self.connection_metadata[websocket]
            
            logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send message to specific connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.active_connections:
            disconnected = []
            message_text = json.dumps(message)
            
            for websocket in self.active_connections:
                try:
                    await websocket.send_text(message_text)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected clients
            for websocket in disconnected:
                self.disconnect(websocket)
    
    async def subscribe_to_stream(self, websocket: WebSocket, stream_id: str):
        """Subscribe WebSocket to specific stream updates"""
        if websocket in self.active_connections:
            self.stream_subscribers[stream_id].add(websocket)
            self.connection_metadata[websocket]["subscribed_streams"].add(stream_id)
            
            await self.send_personal_message(websocket, {
                "type": "subscription_confirmed",
                "stream_id": stream_id,
                "message": f"Subscribed to stream {stream_id}"
            })
            
            logger.info(f"Client subscribed to stream {stream_id}")
    
    async def unsubscribe_from_stream(self, websocket: WebSocket, stream_id: str):
        """Unsubscribe WebSocket from stream updates"""
        if websocket in self.active_connections:
            self.stream_subscribers[stream_id].discard(websocket)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["subscribed_streams"].discard(stream_id)
            
            await self.send_personal_message(websocket, {
                "type": "unsubscription_confirmed",
                "stream_id": stream_id,
                "message": f"Unsubscribed from stream {stream_id}"
            })
    
    async def broadcast_to_stream(self, stream_id: str, message: dict):
        """Broadcast message to all subscribers of a specific stream"""
        subscribers = self.stream_subscribers.get(stream_id, set())
        if subscribers:
            disconnected = []
            message_text = json.dumps(message)
            
            for websocket in subscribers:
                try:
                    await websocket.send_text(message_text)
                except Exception as e:
                    logger.error(f"Error broadcasting to stream subscriber: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected clients
            for websocket in disconnected:
                self.disconnect(websocket)
    
    def get_connection_stats(self):
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "stream_subscriptions": {
                stream_id: len(subscribers) 
                for stream_id, subscribers in self.stream_subscribers.items()
            },
            "connection_details": [
                {
                    "connection_id": id(ws),
                    "connected_at": metadata["connected_at"],
                    "subscribed_streams": list(metadata["subscribed_streams"])
                }
                for ws, metadata in self.connection_metadata.items()
            ]
        }