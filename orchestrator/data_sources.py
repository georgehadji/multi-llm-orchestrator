"""
DataSourceIntegration - Connection templates and health checks.
===============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 8, Phase R5 (Retool-inspired).
"""

from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    POSTGRES = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBHOOK = "webhook"
    S3 = "s3"
    KAFKA = "kafka"


@dataclass
class ConnectionTemplate:
    name: str
    source_type: SourceType
    connection_string: str = ""
    host: str = "localhost"
    port: int = 0
    database: str = ""
    username: str = ""
    headers: dict = field(default_factory=dict)
    ssl_enabled: bool = False
    pool_size: int = 5
    health_endpoint: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.source_type.value,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "ssl": self.ssl_enabled,
            "pool_size": self.pool_size,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            source_type=SourceType(d["type"]),
            host=d.get("host", "localhost"),
            port=d.get("port", 0),
            database=d.get("database", ""),
            pool_size=d.get("pool_size", 5),
        )


DEFAULT_TEMPLATES = {
    "postgres": ConnectionTemplate(
        "postgres", SourceType.POSTGRES, host="localhost", port=5432, pool_size=5
    ),
    "mysql": ConnectionTemplate(
        "mysql", SourceType.MYSQL, host="localhost", port=3306, pool_size=5
    ),
    "redis": ConnectionTemplate(
        "redis", SourceType.REDIS, host="localhost", port=6379, pool_size=10
    ),
    "mongodb": ConnectionTemplate(
        "mongodb", SourceType.MONGODB, host="localhost", port=27017, pool_size=5
    ),
    "rest": ConnectionTemplate(
        "rest", SourceType.REST_API, headers={"Content-Type": "application/json"}
    ),
}


class DataSourceManager:
    """Manages data source connections with templates and health checks."""

    def __init__(self):
        self._sources: dict[str, ConnectionTemplate] = {k: v for k, v in DEFAULT_TEMPLATES.items()}

    def register(self, template):
        self._sources[template.name] = template

    def get(self, name):
        return self._sources.get(name)

    def list_templates(self):
        return [s.to_dict() for s in self._sources.values()]

    def generate_docker_compose(self, sources=None):
        """Generate a Docker Compose file for selected sources."""
        names = sources or list(self._sources.keys())[:3]
        services = {}
        for name in names:
            src = self._sources.get(name)
            if not src:
                continue
            if src.source_type == SourceType.POSTGRES:
                services[name] = {
                    "image": "postgres:16",
                    "environment": {
                        "POSTGRES_DB": src.database or name,
                        "POSTGRES_HOST_AUTH_METHOD": "trust",
                    },
                    "ports": [f"{src.port}:5432"],
                }
            elif src.source_type == SourceType.REDIS:
                services[name] = {"image": "redis:7-alpine", "ports": [f"{src.port}:6379"]}
            elif src.source_type == SourceType.MYSQL:
                services[name] = {
                    "image": "mysql:8",
                    "environment": {
                        "MYSQL_ALLOW_EMPTY_PASSWORD": "yes",
                        "MYSQL_DATABASE": src.database or name,
                    },
                    "ports": [f"{src.port}:3306"],
                }
            elif src.source_type == SourceType.MONGODB:
                services[name] = {"image": "mongo:7", "ports": [f"{src.port}:27017"]}
        return {"version": "3.8", "services": services}

    async def health_check(self, name, timeout=5):
        """Check if a data source is reachable."""
        src = self._sources.get(name)
        if not src:
            return False, "Not found"
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((src.host, src.port))
            sock.close()
            return result == 0, f"Port {src.port} {'reachable' if result == 0 else 'unreachable'}"
        except Exception as e:
            return False, str(e)
