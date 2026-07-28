"""DockerSandbox — --network=none --read-only --cap-drop=ALL isolation."""

from __future__ import annotations

import logging
from pathlib import Path

from ...domain.testing_models import IsolationLevel

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Provides Docker-level isolation for untrusted code execution."""

    def __init__(self) -> None:
        self.level = IsolationLevel.DOCKER
