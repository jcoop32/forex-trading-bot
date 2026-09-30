import os
import json
import logging
from datetime import datetime, timezone
import redis

logger = logging.getLogger("trading_bot")

class RedisClient:
    """
    Redis client providing real-time Pub/Sub event broadcasting and caching.
    Falls back gracefully if Redis is not configured or unavailable.
    """
    def __init__(self, redis_url=None):
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.client = None
        self._connect()

    def _connect(self):
        if not self.redis_url:
            logger.info("Redis: REDIS_URL not configured. Running in stand-alone mode (no Redis).")
            return

        try:
            self.client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0
            )
            self.client.ping()
            logger.info(f"Redis: Connected successfully to {self.redis_url.split('@')[-1]}")
        except Exception as e:
            logger.warning(f"Redis: Connection failed ({e}). Operating in fallback mode.")
            self.client = None

    def is_connected(self):
        return self.client is not None

    def publish_event(self, channel, payload):
        """
        Publish an event to a Redis channel (e.g., 'forex:events', 'forex:alerts').
        """
        if not self.client:
            return False

        try:
            if isinstance(payload, dict):
                if "timestamp" not in payload:
                    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
                message = json.dumps(payload)
            else:
                message = str(payload)

            self.client.publish(channel, message)
            logger.debug(f"Redis: Published to {channel}: {message[:120]}")
            return True
        except Exception as e:
            logger.error(f"Redis publish error on {channel}: {e}")
            return False

    def cache_set(self, key, value, ttl_seconds=5):
        """Cache data with expiration TTL."""
        if not self.client:
            return False
        try:
            val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            self.client.setex(key, ttl_seconds, val_str)
            return True
        except Exception as e:
            logger.error(f"Redis cache set error for {key}: {e}")
            return False

    def cache_get(self, key):
        """Get cached data."""
        if not self.client:
            return None
        try:
            data = self.client.get(key)
            if data:
                try:
                    return json.loads(data)
                except Exception:
                    return data
            return None
        except Exception as e:
            logger.error(f"Redis cache get error for {key}: {e}")
            return None
