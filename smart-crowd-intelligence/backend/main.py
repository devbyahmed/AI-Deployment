"""
FastAPI Backend for Smart Crowd Intelligence System
Real-time crowd analysis API with WebSocket support
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepstream_core.pipeline_config import MultiStreamManager
from deepstream_core.crowd_detector import CrowdAnalyzer
from deepstream_core.movement_tracker import MovementTracker
from deepstream_core.density_calculator import CrowdDensityAnalyzer
from deepstream_core.alert_system import AlertSystem, AlertLevel, AlertType
from ai_models.crowd_detection_model import CrowdModelInference
from ai_models.density_estimation_model import DensityEstimator
from ai_models.behavior_analysis_model import CrowdBehaviorAnalyzer
from backend.websocket_manager import ConnectionManager
from backend.data_processor import DataProcessor
from backend.database import DatabaseManager
from backend.redis_cache import RedisCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Crowd Intelligence API",
    description="Real-time crowd analysis and monitoring system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
stream_manager = MultiStreamManager()
crowd_analyzer = CrowdAnalyzer()
movement_tracker = MovementTracker()
density_analyzer = CrowdDensityAnalyzer()
alert_system = AlertSystem()
behavior_analyzer = CrowdBehaviorAnalyzer()
websocket_manager = ConnectionManager()
data_processor = DataProcessor()
db_manager = DatabaseManager()
redis_cache = RedisCache()

# AI Models
crowd_model = None
density_model = None

# Processing state
processing_active = False
stream_configs = {}

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global crowd_model, density_model
    
    logger.info("Starting Smart Crowd Intelligence System...")
    
    # Initialize database
    await db_manager.initialize()
    
    # Initialize Redis cache
    await redis_cache.initialize()
    
    # Initialize AI models
    try:
        crowd_model = CrowdModelInference("ai_models/crowd_model.pth")
        density_model = DensityEstimator()
        logger.info("AI models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load AI models: {e}")
        # Use fallback models
        crowd_model = CrowdModelInference(None)  # Will use default
        density_model = DensityEstimator()
    
    # Start alert system
    alert_system.start()
    
    # Subscribe to alerts for WebSocket broadcasting
    alert_system.subscribe(broadcast_alert)
    
    logger.info("System initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Smart Crowd Intelligence System...")
    
    # Stop processing
    global processing_active
    processing_active = False
    
    # Stop alert system
    alert_system.stop()
    
    # Close database connections
    await db_manager.close()
    
    # Close Redis connections
    await redis_cache.close()
    
    logger.info("System shutdown complete")

async def broadcast_alert(alert):
    """Broadcast alert to all connected WebSocket clients"""
    await websocket_manager.broadcast({
        "type": "alert",
        "data": alert.to_dict()
    })

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "subscribe":
                # Client subscribing to specific stream
                stream_id = message.get("stream_id")
                await websocket_manager.subscribe_to_stream(websocket, stream_id)
                
            elif message.get("type") == "command":
                # Handle real-time commands
                await handle_websocket_command(websocket, message)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

async def handle_websocket_command(websocket: WebSocket, message: dict):
    """Handle WebSocket commands from clients"""
    command = message.get("command")
    
    if command == "get_status":
        status = await get_system_status()
        await websocket.send_text(json.dumps({
            "type": "status",
            "data": status
        }))
    
    elif command == "acknowledge_alert":
        alert_id = message.get("alert_id")
        user = message.get("user", "websocket_user")
        success = alert_system.acknowledge_alert(alert_id, user)
        
        await websocket.send_text(json.dumps({
            "type": "alert_acknowledged",
            "data": {"alert_id": alert_id, "success": success}
        }))

# REST API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Crowd Intelligence System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": await db_manager.health_check(),
            "redis": await redis_cache.health_check(),
            "alert_system": alert_system.running,
            "ai_models": crowd_model is not None and density_model is not None
        }
    }

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    # Get active streams
    active_streams = len(stream_manager.streams)
    
    # Get recent alerts
    recent_alerts = alert_system.get_active_alerts()
    
    # Get processing statistics
    stats = await redis_cache.get("system_stats") or {}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "processing_active": processing_active,
            "active_streams": active_streams,
            "connected_clients": len(websocket_manager.active_connections)
        },
        "alerts": {
            "active_count": len(recent_alerts),
            "critical_count": len([a for a in recent_alerts if a["level"] == "CRITICAL"]),
            "recent_alerts": recent_alerts[:5]  # Last 5 alerts
        },
        "performance": stats,
        "streams": {
            stream_id: {
                "location": stream["location"],
                "active": stream["active"],
                "crowd_data": stream["crowd_data"]
            }
            for stream_id, stream in stream_manager.streams.items()
        }
    }

# Stream Management Endpoints

@app.post("/streams")
async def add_stream(stream_config: dict):
    """Add a new video stream"""
    try:
        stream_id = stream_config.get("stream_id")
        uri = stream_config.get("uri")
        location = stream_config.get("location", "Unknown")
        
        if not stream_id or not uri:
            raise HTTPException(status_code=400, detail="stream_id and uri are required")
        
        # Add to stream manager
        stream_manager.add_stream(stream_id, uri, location)
        
        # Store configuration
        stream_configs[stream_id] = stream_config
        
        # Cache in Redis
        await redis_cache.set(f"stream_config:{stream_id}", stream_config)
        
        # Store in database
        await db_manager.save_stream_config(stream_config)
        
        logger.info(f"Added stream {stream_id} at {location}")
        
        return {
            "success": True,
            "message": f"Stream {stream_id} added successfully",
            "stream_id": stream_id
        }
        
    except Exception as e:
        logger.error(f"Error adding stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/streams")
async def get_streams():
    """Get all configured streams"""
    streams = []
    for stream_id, stream in stream_manager.streams.items():
        config = stream_configs.get(stream_id, {})
        streams.append({
            "stream_id": stream_id,
            "uri": stream["uri"],
            "location": stream["location"],
            "active": stream["active"],
            "crowd_data": stream["crowd_data"],
            "config": config
        })
    
    return {"streams": streams}

@app.get("/streams/{stream_id}")
async def get_stream(stream_id: str):
    """Get specific stream information"""
    if stream_id not in stream_manager.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    stream = stream_manager.streams[stream_id]
    config = stream_configs.get(stream_id, {})
    
    # Get recent analytics from cache
    recent_data = await redis_cache.get(f"stream_analytics:{stream_id}")
    
    return {
        "stream_id": stream_id,
        "uri": stream["uri"],
        "location": stream["location"],
        "active": stream["active"],
        "crowd_data": stream["crowd_data"],
        "config": config,
        "recent_analytics": recent_data
    }

@app.delete("/streams/{stream_id}")
async def remove_stream(stream_id: str):
    """Remove a video stream"""
    if stream_id not in stream_manager.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Remove from stream manager
    stream_manager.remove_stream(stream_id)
    
    # Remove from configurations
    if stream_id in stream_configs:
        del stream_configs[stream_id]
    
    # Remove from cache
    await redis_cache.delete(f"stream_config:{stream_id}")
    await redis_cache.delete(f"stream_analytics:{stream_id}")
    
    logger.info(f"Removed stream {stream_id}")
    
    return {
        "success": True,
        "message": f"Stream {stream_id} removed successfully"
    }

# Analysis Endpoints

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze a single image for crowd metrics"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Perform analysis
        start_time = time.time()
        
        # Crowd detection
        crowd_result = crowd_model.predict(image)
        
        # Density estimation
        density_result = density_model.estimate_density(image)
        
        # Movement analysis (placeholder for single image)
        movement_result = {
            "avg_speed": 0,
            "movement_directions": [0] * 8,
            "stationary_count": crowd_result.get("count", 0),
            "moving_count": 0
        }
        
        # Combine results
        analysis = {
            "timestamp": time.time(),
            "processing_time": time.time() - start_time,
            "crowd_detection": crowd_result,
            "density_estimation": density_result,
            "movement_analysis": movement_result,
            "image_info": {
                "filename": file.filename,
                "size": len(contents),
                "dimensions": image.size
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/behavior")
async def analyze_behavior(crowd_data: dict):
    """Analyze crowd behavior from crowd data"""
    try:
        # Perform behavior analysis
        behavior_result = behavior_analyzer.analyze_behavior(crowd_data)
        
        return {
            "success": True,
            "behavior_analysis": behavior_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing behavior: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/{stream_id}")
async def get_stream_analytics(stream_id: str, hours: int = 1):
    """Get analytics for a specific stream"""
    try:
        # Get data from database
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        analytics = await db_manager.get_stream_analytics(stream_id, start_time, end_time)
        
        return {
            "stream_id": stream_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "analytics": analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert Management Endpoints

@app.get("/alerts")
async def get_alerts(level: Optional[str] = None, limit: int = 50):
    """Get active alerts"""
    alert_level = None
    if level:
        try:
            alert_level = AlertLevel[level.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail="Invalid alert level")
    
    alerts = alert_system.get_active_alerts(level=alert_level)
    
    return {
        "alerts": alerts[:limit],
        "total_count": len(alerts),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user_info: dict):
    """Acknowledge an alert"""
    user = user_info.get("user", "api_user")
    success = alert_system.acknowledge_alert(alert_id, user)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Broadcast acknowledgment
    await websocket_manager.broadcast({
        "type": "alert_acknowledged",
        "data": {"alert_id": alert_id, "user": user}
    })
    
    return {
        "success": True,
        "message": f"Alert {alert_id} acknowledged by {user}"
    }

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolution_info: dict):
    """Resolve an alert"""
    user = resolution_info.get("user", "api_user")
    note = resolution_info.get("note", "")
    
    success = alert_system.resolve_alert(alert_id, user, note)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Broadcast resolution
    await websocket_manager.broadcast({
        "type": "alert_resolved",
        "data": {"alert_id": alert_id, "user": user, "note": note}
    })
    
    return {
        "success": True,
        "message": f"Alert {alert_id} resolved by {user}"
    }

@app.post("/alerts/manual")
async def create_manual_alert(alert_data: dict):
    """Create a manual alert"""
    try:
        alert_type = AlertType(alert_data.get("type"))
        alert_level = AlertLevel[alert_data.get("level").upper()]
        message = alert_data.get("message")
        stream_id = alert_data.get("stream_id")
        location = alert_data.get("location")
        user = alert_data.get("user", "api_user")
        
        alert_id = alert_system.create_manual_alert(
            alert_type, alert_level, message, stream_id, location, 
            data=alert_data.get("data"), user=user
        )
        
        return {
            "success": True,
            "alert_id": alert_id,
            "message": "Manual alert created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating manual alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/statistics")
async def get_alert_statistics(hours: int = 24):
    """Get alert statistics"""
    time_window = hours * 3600  # Convert to seconds
    stats = alert_system.get_alert_statistics(time_window)
    
    return {
        "time_window_hours": hours,
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }

# Configuration Endpoints

@app.get("/config")
async def get_configuration():
    """Get system configuration"""
    config = {
        "alert_thresholds": alert_system.thresholds,
        "processing_settings": {
            "active": processing_active,
            "frame_rate": 30,
            "batch_size": 4
        },
        "model_settings": {
            "crowd_model_loaded": crowd_model is not None,
            "density_model_loaded": density_model is not None,
            "behavior_model_loaded": True
        }
    }
    
    return config

@app.put("/config")
async def update_configuration(config_update: dict):
    """Update system configuration"""
    try:
        # Update alert thresholds
        if "alert_thresholds" in config_update:
            alert_system.thresholds.update(config_update["alert_thresholds"])
        
        # Update other settings as needed
        # This would include updating processing parameters, model settings, etc.
        
        # Cache updated configuration
        await redis_cache.set("system_config", config_update)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Export Endpoints

@app.get("/export/analytics")
async def export_analytics(
    stream_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    format: str = "json"
):
    """Export analytics data"""
    try:
        # Parse dates
        if start_date:
            start_time = datetime.fromisoformat(start_date)
        else:
            start_time = datetime.now() - timedelta(days=1)
        
        if end_date:
            end_time = datetime.fromisoformat(end_date)
        else:
            end_time = datetime.now()
        
        # Get data from database
        data = await db_manager.export_analytics(stream_id, start_time, end_time)
        
        if format.lower() == "csv":
            # Convert to CSV format
            csv_data = data_processor.to_csv(data)
            return StreamingResponse(
                io.StringIO(csv_data),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=analytics.csv"}
            )
        else:
            return {
                "data": data,
                "metadata": {
                    "stream_id": stream_id,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "record_count": len(data)
                }
            }
            
    except Exception as e:
        logger.error(f"Error exporting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background processing task
async def process_video_streams():
    """Background task to process video streams"""
    global processing_active
    
    while processing_active:
        try:
            # Process each active stream
            for stream_id, stream in stream_manager.streams.items():
                if stream["active"]:
                    # This would integrate with actual video processing
                    # For now, we'll simulate analysis
                    
                    # Generate simulated crowd data
                    simulated_data = generate_simulated_crowd_data(stream_id)
                    
                    # Update stream data
                    stream_manager.update_crowd_data(stream_id, simulated_data)
                    
                    # Perform behavior analysis
                    behavior_result = behavior_analyzer.analyze_behavior({
                        "density": simulated_data,
                        "movement": simulated_data,
                        "tracks": []
                    })
                    
                    # Check for alerts
                    alerts = alert_system.analyze_crowd_data(
                        {"density": simulated_data, "movement": simulated_data},
                        stream_id,
                        stream["location"]
                    )
                    
                    # Cache analytics
                    analytics_data = {
                        "timestamp": time.time(),
                        "crowd_data": simulated_data,
                        "behavior_analysis": behavior_result,
                        "alerts": [alert.to_dict() for alert in alerts]
                    }
                    
                    await redis_cache.set(f"stream_analytics:{stream_id}", analytics_data, expire=3600)
                    
                    # Broadcast to WebSocket clients
                    await websocket_manager.broadcast_to_stream(stream_id, {
                        "type": "analytics_update",
                        "stream_id": stream_id,
                        "data": analytics_data
                    })
            
            # Update system statistics
            stats = {
                "last_update": time.time(),
                "streams_processed": len([s for s in stream_manager.streams.values() if s["active"]]),
                "total_alerts": len(alert_system.get_active_alerts())
            }
            await redis_cache.set("system_stats", stats)
            
            # Wait before next processing cycle
            await asyncio.sleep(1.0)  # Process at 1 Hz
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            await asyncio.sleep(5.0)  # Wait longer on error

def generate_simulated_crowd_data(stream_id: str) -> dict:
    """Generate simulated crowd data for testing"""
    import random
    
    base_density = random.uniform(5, 25)
    base_speed = random.uniform(2, 12)
    
    return {
        "overall_density": base_density + random.uniform(-2, 2),
        "max_local_density": base_density * 1.5,
        "avg_local_density": base_density * 0.8,
        "density_variance": random.uniform(0.5, 3.0),
        "hotspots": [
            {
                "density": random.uniform(10, 30),
                "coordinates": [
                    random.randint(100, 500),
                    random.randint(100, 400),
                    random.randint(500, 800),
                    random.randint(400, 600)
                ],
                "severity": random.choice(["medium", "high"])
            }
        ] if random.random() > 0.7 else [],
        "avg_speed": base_speed + random.uniform(-1, 1),
        "max_speed": base_speed * 1.8,
        "movement_ratio": random.uniform(0.3, 0.9),
        "flow_consistency": random.uniform(0.4, 1.0),
        "congestion_level": random.choice(["low", "medium", "high"]),
        "density_trend": random.choice(["stable", "increasing", "decreasing"]),
        "trend_strength": random.uniform(0, 0.5)
    }

# Start background processing
@app.on_event("startup")
async def start_background_processing():
    """Start background video processing"""
    global processing_active
    processing_active = True
    asyncio.create_task(process_video_streams())

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )