"""
Redis Cache Manager
High-performance caching for real-time crowd intelligence data
"""

import aioredis
import json
import logging
import pickle
from typing import Any, Optional, Dict, List
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis cache manager for real-time data"""
    
    def __init__(self):
        self.redis = None
        self.connection_params = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "db": int(os.getenv("REDIS_DB", 0)),
            "password": os.getenv("REDIS_PASSWORD"),
            "decode_responses": True
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            # Remove None password to avoid issues
            params = {k: v for k, v in self.connection_params.items() if v is not None}
            
            self.redis = await aioredis.create_redis_pool(
                f"redis://{params['host']}:{params['port']}/{params['db']}",
                password=params.get('password'),
                encoding='utf-8'
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis = None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None):
        """Set value in cache with optional expiration"""
        if not self.redis:
            return False
            
        try:
            # Serialize complex objects
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            if expire:
                await self.redis.setex(key, expire, serialized_value)
            else:
                await self.redis.set(key, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        if not self.redis:
            return default
            
        try:
            value = await self.redis.get(key)
            if value is None:
                return default
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return default
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis:
            return False
            
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis:
            return False
            
        try:
            return await self.redis.exists(key)
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value"""
        if not self.redis:
            return 0
            
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for existing key"""
        if not self.redis:
            return False
            
        try:
            return await self.redis.expire(key, seconds)
        except Exception as e:
            logger.error(f"Error setting expiration for key {key}: {e}")
            return False
    
    async def set_hash(self, key: str, field: str, value: Any):
        """Set hash field value"""
        if not self.redis:
            return False
            
        try:
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
                
            await self.redis.hset(key, field, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Error setting hash {key}:{field}: {e}")
            return False
    
    async def get_hash(self, key: str, field: str, default: Any = None) -> Any:
        """Get hash field value"""
        if not self.redis:
            return default
            
        try:
            value = await self.redis.hget(key, field)
            if value is None:
                return default
                
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.error(f"Error getting hash {key}:{field}: {e}")
            return default
    
    async def get_all_hash(self, key: str) -> Dict[str, Any]:
        """Get all hash fields"""
        if not self.redis:
            return {}
            
        try:
            hash_data = await self.redis.hgetall(key)
            result = {}
            
            for field, value in hash_data.items():
                try:
                    result[field] = json.loads(value)
                except json.JSONDecodeError:
                    result[field] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting all hash {key}: {e}")
            return {}
    
    async def delete_hash_field(self, key: str, field: str) -> bool:
        """Delete hash field"""
        if not self.redis:
            return False
            
        try:
            result = await self.redis.hdel(key, field)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting hash field {key}:{field}: {e}")
            return False
    
    async def push_to_list(self, key: str, value: Any, max_length: Optional[int] = None):
        """Push value to list (left push)"""
        if not self.redis:
            return False
            
        try:
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            await self.redis.lpush(key, serialized_value)
            
            # Trim list if max_length specified
            if max_length:
                await self.redis.ltrim(key, 0, max_length - 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return False
    
    async def get_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get list range"""
        if not self.redis:
            return []
            
        try:
            values = await self.redis.lrange(key, start, end)
            result = []
            
            for value in values:
                try:
                    result.append(json.loads(value))
                except json.JSONDecodeError:
                    result.append(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting list {key}: {e}")
            return []
    
    async def get_list_length(self, key: str) -> int:
        """Get list length"""
        if not self.redis:
            return 0
            
        try:
            return await self.redis.llen(key)
        except Exception as e:
            logger.error(f"Error getting list length {key}: {e}")
            return 0
    
    async def add_to_set(self, key: str, value: Any) -> bool:
        """Add value to set"""
        if not self.redis:
            return False
            
        try:
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
                
            result = await self.redis.sadd(key, serialized_value)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error adding to set {key}: {e}")
            return False
    
    async def get_set_members(self, key: str) -> List[Any]:
        """Get all set members"""
        if not self.redis:
            return []
            
        try:
            members = await self.redis.smembers(key)
            result = []
            
            for member in members:
                try:
                    result.append(json.loads(member))
                except json.JSONDecodeError:
                    result.append(member)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting set members {key}: {e}")
            return []
    
    async def is_set_member(self, key: str, value: Any) -> bool:
        """Check if value is in set"""
        if not self.redis:
            return False
            
        try:
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
                
            return await self.redis.sismember(key, serialized_value)
            
        except Exception as e:
            logger.error(f"Error checking set membership {key}: {e}")
            return False
    
    async def cache_stream_data(self, stream_id: str, data: Dict, expire: int = 3600):
        """Cache stream-specific data"""
        cache_key = f"stream_data:{stream_id}"
        await self.set(cache_key, data, expire)
    
    async def get_cached_stream_data(self, stream_id: str) -> Optional[Dict]:
        """Get cached stream data"""
        cache_key = f"stream_data:{stream_id}"
        return await self.get(cache_key)
    
    async def cache_analytics_batch(self, analytics_batch: List[Dict]):
        """Cache batch of analytics data"""
        for analytics in analytics_batch:
            stream_id = analytics.get("stream_id")
            timestamp = analytics.get("timestamp", datetime.now().timestamp())
            
            # Cache by stream and time
            cache_key = f"analytics:{stream_id}:{int(timestamp)}"
            await self.set(cache_key, analytics, expire=86400)  # 24 hours
            
            # Add to stream analytics list
            list_key = f"analytics_list:{stream_id}"
            await self.push_to_list(list_key, analytics, max_length=1000)
    
    async def get_recent_analytics(self, stream_id: str, limit: int = 100) -> List[Dict]:
        """Get recent analytics for stream"""
        list_key = f"analytics_list:{stream_id}"
        return await self.get_list(list_key, 0, limit - 1)
    
    async def cache_alert(self, alert_data: Dict):
        """Cache alert data"""
        alert_id = alert_data.get("id")
        stream_id = alert_data.get("stream_id")
        
        # Cache alert by ID
        await self.set(f"alert:{alert_id}", alert_data, expire=86400)
        
        # Add to stream alerts
        if stream_id:
            await self.push_to_list(f"stream_alerts:{stream_id}", alert_data, max_length=100)
        
        # Add to global active alerts
        if not alert_data.get("resolved", False):
            await self.add_to_set("active_alerts", alert_id)
    
    async def get_active_alerts(self) -> List[str]:
        """Get active alert IDs"""
        return await self.get_set_members("active_alerts")
    
    async def resolve_alert_cache(self, alert_id: str):
        """Mark alert as resolved in cache"""
        # Remove from active alerts
        await self.redis.srem("active_alerts", alert_id)
        
        # Update alert data
        alert_data = await self.get(f"alert:{alert_id}")
        if alert_data:
            alert_data["resolved"] = True
            alert_data["resolution_time"] = datetime.now().timestamp()
            await self.set(f"alert:{alert_id}", alert_data, expire=86400)
    
    async def get_system_metrics(self) -> Dict:
        """Get cached system performance metrics"""
        metrics = {}
        
        # Get various cached metrics
        metrics["memory_usage"] = await self.get("metrics:memory_usage", 0)
        metrics["cpu_usage"] = await self.get("metrics:cpu_usage", 0)
        metrics["processing_fps"] = await self.get("metrics:processing_fps", 0)
        metrics["active_streams"] = await self.get("metrics:active_streams", 0)
        metrics["total_alerts"] = await self.get("metrics:total_alerts", 0)
        
        return metrics
    
    async def update_system_metrics(self, metrics: Dict):
        """Update system performance metrics"""
        for key, value in metrics.items():
            await self.set(f"metrics:{key}", value, expire=300)  # 5 minutes
    
    async def cleanup_expired_data(self):
        """Clean up expired cache entries"""
        if not self.redis:
            return
            
        try:
            # Get keys that might be expired
            current_time = datetime.now().timestamp()
            
            # Clean up old analytics (older than 24 hours)
            pattern = "analytics:*"
            keys = await self.redis.keys(pattern)
            
            for key in keys:
                # Extract timestamp from key
                try:
                    timestamp = int(key.split(":")[-1])
                    if current_time - timestamp > 86400:  # 24 hours
                        await self.delete(key)
                except (ValueError, IndexError):
                    continue
            
            logger.info("Cache cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    async def get_cache_statistics(self) -> Dict:
        """Get cache usage statistics"""
        if not self.redis:
            return {}
            
        try:
            info = await self.redis.info()
            
            return {
                "memory_used": info.get("used_memory_human", "0B"),
                "memory_peak": info.get("used_memory_peak_human", "0B"),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "expired_keys": info.get("expired_keys", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "connected_clients": info.get("connected_clients", 0),
                "uptime": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        if not self.redis:
            return False
            
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connections"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            logger.info("Redis connections closed")