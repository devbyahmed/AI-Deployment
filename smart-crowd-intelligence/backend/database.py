"""
Database Manager
PostgreSQL database operations for crowd intelligence data
"""

import asyncpg
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database operations"""
    
    def __init__(self):
        self.pool = None
        self.connection_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", 5432),
            "database": os.getenv("DB_NAME", "crowd_intelligence"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password")
        }
    
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                **self.connection_params,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Create tables
            await self.create_tables()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Create fallback in-memory storage
            self.pool = None
    
    async def create_tables(self):
        """Create necessary database tables"""
        if not self.pool:
            return
            
        async with self.pool.acquire() as conn:
            # Streams table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS streams (
                    stream_id VARCHAR(255) PRIMARY KEY,
                    uri TEXT NOT NULL,
                    location VARCHAR(255),
                    config JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Analytics data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id SERIAL PRIMARY KEY,
                    stream_id VARCHAR(255) REFERENCES streams(stream_id),
                    timestamp TIMESTAMP NOT NULL,
                    crowd_data JSONB,
                    density_data JSONB,
                    movement_data JSONB,
                    behavior_data JSONB,
                    processing_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id VARCHAR(255) PRIMARY KEY,
                    stream_id VARCHAR(255),
                    alert_type VARCHAR(100),
                    alert_level VARCHAR(50),
                    message TEXT,
                    data JSONB,
                    timestamp TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    acknowledged_by VARCHAR(255),
                    resolved_by VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Events table for significant occurrences
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    stream_id VARCHAR(255),
                    event_type VARCHAR(100),
                    event_data JSONB,
                    timestamp TIMESTAMP,
                    severity VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # System logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    log_level VARCHAR(50),
                    message TEXT,
                    component VARCHAR(100),
                    data JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_stream_timestamp ON analytics(stream_id, timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_stream_timestamp ON alerts(stream_id, timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_stream_timestamp ON events(stream_id, timestamp)")
    
    async def save_stream_config(self, stream_config: Dict):
        """Save stream configuration"""
        if not self.pool:
            return
            
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO streams (stream_id, uri, location, config, updated_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                ON CONFLICT (stream_id) 
                DO UPDATE SET 
                    uri = EXCLUDED.uri,
                    location = EXCLUDED.location,
                    config = EXCLUDED.config,
                    updated_at = CURRENT_TIMESTAMP
            """, 
            stream_config["stream_id"],
            stream_config["uri"],
            stream_config.get("location"),
            json.dumps(stream_config)
            )
    
    async def save_analytics_data(self, stream_id: str, analytics_data: Dict):
        """Save analytics data"""
        if not self.pool:
            return
            
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO analytics 
                (stream_id, timestamp, crowd_data, density_data, movement_data, behavior_data, processing_time)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            stream_id,
            datetime.fromtimestamp(analytics_data.get("timestamp", datetime.now().timestamp())),
            json.dumps(analytics_data.get("crowd_data", {})),
            json.dumps(analytics_data.get("density_data", {})),
            json.dumps(analytics_data.get("movement_data", {})),
            json.dumps(analytics_data.get("behavior_data", {})),
            analytics_data.get("processing_time", 0.0)
            )
    
    async def save_alert(self, alert_data: Dict):
        """Save alert to database"""
        if not self.pool:
            return
            
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO alerts 
                (id, stream_id, alert_type, alert_level, message, data, timestamp, acknowledged, resolved)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    acknowledged = EXCLUDED.acknowledged,
                    resolved = EXCLUDED.resolved,
                    acknowledged_by = EXCLUDED.acknowledged_by,
                    resolved_by = EXCLUDED.resolved_by
            """,
            alert_data["id"],
            alert_data.get("stream_id"),
            alert_data.get("type"),
            alert_data.get("level"),
            alert_data.get("message"),
            json.dumps(alert_data.get("data", {})),
            datetime.fromtimestamp(alert_data.get("timestamp", datetime.now().timestamp())),
            alert_data.get("acknowledged", False),
            alert_data.get("resolved", False)
            )
    
    async def get_stream_analytics(self, stream_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get analytics data for a stream within time range"""
        if not self.pool:
            return []
            
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM analytics 
                WHERE stream_id = $1 AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp DESC
            """, stream_id, start_time, end_time)
            
            return [dict(row) for row in rows]
    
    async def get_alerts(self, stream_id: Optional[str] = None, hours: int = 24) -> List[Dict]:
        """Get alerts within time range"""
        if not self.pool:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        async with self.pool.acquire() as conn:
            if stream_id:
                rows = await conn.fetch("""
                    SELECT * FROM alerts 
                    WHERE stream_id = $1 AND timestamp > $2
                    ORDER BY timestamp DESC
                """, stream_id, cutoff_time)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM alerts 
                    WHERE timestamp > $1
                    ORDER BY timestamp DESC
                """, cutoff_time)
            
            return [dict(row) for row in rows]
    
    async def export_analytics(self, stream_id: Optional[str], start_time: datetime, end_time: datetime) -> List[Dict]:
        """Export analytics data for given parameters"""
        if not self.pool:
            return []
            
        async with self.pool.acquire() as conn:
            if stream_id:
                rows = await conn.fetch("""
                    SELECT a.*, s.location, s.uri 
                    FROM analytics a
                    JOIN streams s ON a.stream_id = s.stream_id
                    WHERE a.stream_id = $1 AND a.timestamp BETWEEN $2 AND $3
                    ORDER BY a.timestamp
                """, stream_id, start_time, end_time)
            else:
                rows = await conn.fetch("""
                    SELECT a.*, s.location, s.uri 
                    FROM analytics a
                    JOIN streams s ON a.stream_id = s.stream_id
                    WHERE a.timestamp BETWEEN $1 AND $2
                    ORDER BY a.timestamp
                """, start_time, end_time)
            
            return [dict(row) for row in rows]
    
    async def log_system_event(self, level: str, message: str, component: str, data: Optional[Dict] = None):
        """Log system events"""
        if not self.pool:
            logger.info(f"[{component}] {level}: {message}")
            return
            
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO system_logs (log_level, message, component, data)
                VALUES ($1, $2, $3, $4)
            """, level, message, component, json.dumps(data or {}))
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to manage database size"""
        if not self.pool:
            return
            
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with self.pool.acquire() as conn:
            # Clean up old analytics data
            result = await conn.execute("""
                DELETE FROM analytics WHERE created_at < $1
            """, cutoff_date)
            
            # Clean up old resolved alerts
            await conn.execute("""
                DELETE FROM alerts 
                WHERE resolved = TRUE AND created_at < $1
            """, cutoff_date)
            
            # Clean up old system logs
            await conn.execute("""
                DELETE FROM system_logs WHERE timestamp < $1
            """, cutoff_date)
            
            logger.info(f"Cleaned up data older than {cutoff_date}")
    
    async def get_system_statistics(self) -> Dict:
        """Get system-wide statistics"""
        if not self.pool:
            return {}
            
        async with self.pool.acquire() as conn:
            # Stream statistics
            stream_count = await conn.fetchval("SELECT COUNT(*) FROM streams")
            
            # Analytics statistics
            total_analytics = await conn.fetchval("SELECT COUNT(*) FROM analytics")
            analytics_24h = await conn.fetchval("""
                SELECT COUNT(*) FROM analytics 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            
            # Alert statistics
            total_alerts = await conn.fetchval("SELECT COUNT(*) FROM alerts")
            active_alerts = await conn.fetchval("""
                SELECT COUNT(*) FROM alerts 
                WHERE resolved = FALSE
            """)
            
            # Recent activity
            recent_events = await conn.fetchval("""
                SELECT COUNT(*) FROM events 
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            
            return {
                "streams": {
                    "total": stream_count,
                },
                "analytics": {
                    "total": total_analytics,
                    "last_24h": analytics_24h
                },
                "alerts": {
                    "total": total_alerts,
                    "active": active_alerts
                },
                "activity": {
                    "recent_events": recent_events
                },
                "last_updated": datetime.now().isoformat()
            }
    
    async def health_check(self) -> bool:
        """Check database health"""
        if not self.pool:
            return False
            
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connections closed")