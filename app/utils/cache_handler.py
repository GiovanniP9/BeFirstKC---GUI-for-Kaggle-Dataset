from utils.config_loader import ConfigLoader
import redis
import pickle


def get_redis_client():
    cfg = ConfigLoader.get_redis_config()
    return redis.StrictRedis(host=cfg["host"], port=cfg["port"], db=cfg["db"])

def cache_result(key: str, value, expire_seconds: int = 3600):
    client = get_redis_client()
    client.setex(key, expire_seconds, pickle.dumps(value))

def get_cached_result(key: str):
    client = get_redis_client()
    data = client.get(key)
    return pickle.loads(data) if data else None
