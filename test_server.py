#!/usr/bin/env python3
"""
Simplified Smart Crowd Intelligence Test Server for GitHub Codespaces
This version runs without DeepStream and provides API testing capabilities.
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Crowd Intelligence System - Demo",
    description="Simplified version for GitHub Codespaces testing",
    version="1.0.0"
)

# Data Models
class CrowdData(BaseModel):
    timestamp: str
    stream_id: str
    crowd_count: int
    density_level: str  # low, medium, high, critical
    movement_speed: float
    behavior_analysis: Dict
    alerts: List[str]

# In-memory storage for demo
active_connections: List[WebSocket] = []
recent_data: List[CrowdData] = []

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Simulation Functions
def generate_mock_crowd_data(stream_id: str = "demo_stream") -> CrowdData:
    """Generate realistic mock crowd data for testing."""
    crowd_count = random.randint(10, 300)
    
    # Determine density level based on count
    if crowd_count < 50:
        density = "low"
    elif crowd_count < 150:
        density = "medium"
    elif crowd_count < 250:
        density = "high"
    else:
        density = "critical"
    
    # Generate behavior analysis
    behaviors = {
        "normal": random.uniform(0.6, 0.9),
        "aggressive": random.uniform(0.0, 0.2),
        "panic": random.uniform(0.0, 0.1),
        "gathering": random.uniform(0.1, 0.4),
        "dispersing": random.uniform(0.1, 0.3)
    }
    
    # Generate alerts based on conditions
    alerts = []
    if crowd_count > 200:
        alerts.append("High crowd density detected")
    if behaviors["panic"] > 0.05:
        alerts.append("Panic behavior detected")
    if behaviors["aggressive"] > 0.15:
        alerts.append("Aggressive behavior detected")
    
    return CrowdData(
        timestamp=datetime.now().isoformat(),
        stream_id=stream_id,
        crowd_count=crowd_count,
        density_level=density,
        movement_speed=random.uniform(0.5, 3.0),
        behavior_analysis=behaviors,
        alerts=alerts
    )

# Background task for data simulation
async def simulate_data_stream():
    """Continuously generate and broadcast mock data."""
    while True:
        try:
            # Generate new crowd data
            crowd_data = generate_mock_crowd_data()
            recent_data.append(crowd_data)
            
            # Keep only last 100 records
            if len(recent_data) > 100:
                recent_data.pop(0)
            
            # Broadcast to all connected clients
            broadcast_data = {
                "type": "stream_update",
                "data": crowd_data.dict()
            }
            await manager.broadcast(json.dumps(broadcast_data))
            
            # Wait before next update
            await asyncio.sleep(3)  # Update every 3 seconds
            
        except Exception as e:
            print(f"Error in data simulation: {e}")
            await asyncio.sleep(5)

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Smart Crowd Intelligence - Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: white; }
        .header { text-align: center; margin-bottom: 30px; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #16213e; padding: 20px; border-radius: 10px; border: 1px solid #0f3460; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: #0f3460; border-radius: 5px; }
        .value { font-weight: bold; color: #4ade80; }
        .logs { height: 200px; overflow-y: auto; background: #000; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; }
        .log-entry { margin: 2px 0; color: #10b981; }
        .status-live { background: #10b981; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎥 Smart Crowd Intelligence System</h1>
        <p>Real-time AI-powered crowd analysis</p>
        <span class="status-live">● LIVE DEMO</span>
    </div>

    <div class="dashboard">
        <div class="card">
            <h3>📊 Live Stream Data</h3>
            <div class="metric"><span>Crowd Count:</span><span class="value" id="crowd-count">-</span></div>
            <div class="metric"><span>Density:</span><span class="value" id="density">-</span></div>
            <div class="metric"><span>Movement:</span><span class="value" id="movement">-</span></div>
            <div class="metric"><span>Last Update:</span><span class="value" id="last-update">-</span></div>
        </div>

        <div class="card">
            <h3>🧠 AI Behavior Analysis</h3>
            <div class="metric"><span>Normal:</span><span class="value" id="normal">-</span></div>
            <div class="metric"><span>Aggressive:</span><span class="value" id="aggressive">-</span></div>
            <div class="metric"><span>Panic:</span><span class="value" id="panic">-</span></div>
            <div class="metric"><span>Gathering:</span><span class="value" id="gathering">-</span></div>
        </div>

        <div class="card">
            <h3>🚨 Real-time Alerts</h3>
            <div id="alerts" style="min-height: 100px;">
                <p style="text-align: center; opacity: 0.7;">No active alerts</p>
            </div>
        </div>

        <div class="card">
            <h3>📝 System Logs</h3>
            <div class="logs" id="logs">
                <div class="log-entry">System initialized - Demo mode</div>
                <div class="log-entry">WebSocket connection ready</div>
                <div class="log-entry">AI models loaded (CPU mode)</div>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        const logs = document.getElementById('logs');

        function addLog(message) {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
            logs.appendChild(entry);
            logs.scrollTop = logs.scrollHeight;
        }

        function updateDashboard(data) {
            document.getElementById('crowd-count').textContent = data.crowd_count;
            document.getElementById('density').textContent = data.density_level.toUpperCase();
            document.getElementById('movement').textContent = data.movement_speed.toFixed(1) + ' m/s';
            document.getElementById('last-update').textContent = new Date(data.timestamp).toLocaleTimeString();

            const behaviors = data.behavior_analysis;
            document.getElementById('normal').textContent = (behaviors.normal * 100).toFixed(1) + '%';
            document.getElementById('aggressive').textContent = (behaviors.aggressive * 100).toFixed(1) + '%';
            document.getElementById('panic').textContent = (behaviors.panic * 100).toFixed(1) + '%';
            document.getElementById('gathering').textContent = (behaviors.gathering * 100).toFixed(1) + '%';

            const alertsDiv = document.getElementById('alerts');
            if (data.alerts && data.alerts.length > 0) {
                alertsDiv.innerHTML = data.alerts.map(alert => 
                    `<div style="background: #dc2626; padding: 8px; margin: 5px 0; border-radius: 5px;">⚠️ ${alert}</div>`
                ).join('');
            } else {
                alertsDiv.innerHTML = '<p style="text-align: center; opacity: 0.7;">No active alerts</p>';
            }
        }

        ws.onopen = () => addLog('Connected to Smart Crowd Intelligence System');
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'stream_update') {
                updateDashboard(message.data);
                addLog(`Update: ${message.data.crowd_count} people detected`);
            }
        };
        ws.onclose = () => addLog('Connection closed');
        ws.onerror = () => addLog('Connection error');
    </script>
</body>
</html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "mode": "demo",
        "active_connections": len(manager.active_connections),
        "data_points": len(recent_data)
    }

@app.get("/analytics")
async def get_analytics():
    """Get analytics data."""
    if not recent_data:
        return {"message": "No data available yet"}
    
    avg_crowd = sum(d.crowd_count for d in recent_data) / len(recent_data)
    max_crowd = max(d.crowd_count for d in recent_data)
    
    return {
        "total_data_points": len(recent_data),
        "average_crowd_count": round(avg_crowd, 2),
        "max_crowd_count": max_crowd,
        "recent_data": recent_data[-5:]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    print("🚀 Smart Crowd Intelligence System starting...")
    print("📡 Demo mode enabled - generating simulated data")
    asyncio.create_task(simulate_data_stream())
    print("✅ System ready at http://localhost:8000")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")