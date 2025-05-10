import yaml
import os

class ConfigLoader:
    _config = None

    @classmethod
    def load_config(cls, path="config/config.yaml"):
        if cls._config is None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, "r") as f:
                cls._config = yaml.safe_load(f)
        return cls._config

    @classmethod
    def get_mysql_config(cls):
        return cls.load_config().get("mysql", {})

    @classmethod
    def get_redis_config(cls):
        return cls.load_config().get("redis", {})

    @classmethod
    def get_app_config(cls):
        return cls.load_config().get("app", {})
    