"""NexusScope configuration."""
import os
from dataclasses import dataclass


@dataclass
class NexusScopeConfig:
    """Configuration for the NexusScope profiler."""
    enabled: bool = False
    interval: float = 0.001
    async_mode: bool = True
    buffer_size: int = 100

    @classmethod
    def from_env(cls) -> "NexusScopeConfig":
        return cls(
            enabled=os.getenv("ORCHESTRATOR_PROFILING", "0") == "1",
            interval=float(os.getenv("NEXUSSCOPE_INTERVAL", "0.001")),
            async_mode=os.getenv("NEXUSSCOPE_ASYNC_MODE", "1") == "1",
            buffer_size=int(os.getenv("NEXUSSCOPE_BUFFER_SIZE", "100")),
        )
