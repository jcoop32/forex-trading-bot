import unittest
from unittest.mock import MagicMock, patch
from redis_client import RedisClient

class TestRedisClient(unittest.TestCase):
    def test_fallback_when_no_redis_url(self):
        with patch.dict('os.environ', {}, clear=True):
            client = RedisClient(redis_url=None)
            self.assertFalse(client.is_connected())
            # Safe no-ops
            self.assertFalse(client.publish_event("test", {"msg": "hello"}))
            self.assertFalse(client.cache_set("key", "val"))
            self.assertIsNone(client.cache_get("key"))

    def test_mocked_redis_connection_and_publish(self):
        with patch('redis.Redis.from_url') as mock_from_url:
            mock_instance = MagicMock()
            mock_from_url.return_value = mock_instance
            mock_instance.ping.return_value = True

            client = RedisClient(redis_url="redis://localhost:6379/0")
            self.assertTrue(client.is_connected())

            # Test publish event
            res = client.publish_event("forex:events", {"event": "TEST", "data": 123})
            self.assertTrue(res)
            mock_instance.publish.assert_called_once()
            call_args = mock_instance.publish.call_args[0]
            self.assertEqual(call_args[0], "forex:events")
            self.assertIn('"event": "TEST"', call_args[1])

    def test_mocked_cache_operations(self):
        with patch('redis.Redis.from_url') as mock_from_url:
            mock_instance = MagicMock()
            mock_from_url.return_value = mock_instance
            mock_instance.ping.return_value = True
            mock_instance.get.return_value = '{"cached": 100}'

            client = RedisClient(redis_url="redis://localhost:6379/0")
            
            # Cache Set
            set_res = client.cache_set("test_key", {"balance": 1000}, ttl_seconds=10)
            self.assertTrue(set_res)
            mock_instance.setex.assert_called_once()

            # Cache Get
            get_res = client.cache_get("test_key")
            self.assertEqual(get_res, {"cached": 100})

if __name__ == '__main__':
    unittest.main()
