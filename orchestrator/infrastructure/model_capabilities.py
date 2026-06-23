"""OpenRouter model capability map — structured-output gating.

Infrastructure adapter. Fetches the OpenRouter ``/models`` catalogue and exposes
which models advertise ``structured_outputs`` in their ``supported_parameters``,
so the dispatch path only attaches a ``response_format`` JSON schema to models
that actually honor it. Models that lack it (e.g. ``z-ai/glm-4.5-air``,
``minimax/minimax-01``, several ``nvidia/nemotron-*:free`` variants) either
ignore or reject the parameter, so we fall back to prompt-level JSON
instructions plus the response-healing plugin instead.

The pure lookup (``build_capability_map`` / ``supports_structured_output``) is
deliberately separated from the I/O (``get_capability_map`` fetch + disk cache)
so the gating logic stays unit-testable without network access.
"""

from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Mapping

logger = logging.getLogger("orchestrator.api")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_CACHE_PATH = Path.home() / ".orchestrator_cache" / "openrouter_capabilities.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60  # refresh the catalogue at most once a day
_STRUCTURED_OUTPUT_PARAM = "structured_outputs"

# Process-wide memo of {model_id: set(supported_parameters)}; populated lazily.
_capability_map: dict[str, set[str]] | None = None


def _base_slug(model_id: str) -> str:
    """Strip an OpenRouter variant suffix (':free', ':nitro', ':exacto', ...)."""
    return model_id.split(":", 1)[0]


def build_capability_map(models: Iterable[Mapping]) -> dict[str, set[str]]:
    """Build ``{model_id: set(supported_parameters)}`` from a ``/models`` list.

    Pure: accepts the already-decoded ``data`` list, performs no I/O. Indexes by
    both ``id`` and ``canonical_slug`` so alias slugs resolve to the same caps.
    """
    out: dict[str, set[str]] = {}
    for m in models:
        params = set(m.get("supported_parameters") or [])
        for key in (m.get("id"), m.get("canonical_slug")):
            if key:
                out[key] = params
    return out


def supports_structured_output(
    model_id: str, capability_map: Mapping[str, set[str]] | None
) -> bool:
    """Return True if ``model_id`` advertises structured-output support.

    Pure/testable. Resolution order: exact id, then base slug (variant stripped).
    A model that is *present and lacks* ``structured_outputs`` returns False. An
    *unknown* model (absent from the map, or a ``None``/empty map) returns True —
    we only suppress ``response_format`` for models we positively know reject it,
    so a stale or unavailable catalogue never silently disables schemas.
    """
    if not capability_map:
        return True
    for key in (model_id, _base_slug(model_id)):
        params = capability_map.get(key)
        if params is not None:
            return _STRUCTURED_OUTPUT_PARAM in params
    return True


def _fetch_models(timeout: float = 30.0) -> list[dict]:
    """Fetch the raw ``data`` list from the OpenRouter models API."""
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(  # noqa: S310 - fixed https OpenRouter endpoint
        OPENROUTER_MODELS_URL, context=ctx, timeout=timeout
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("data", [])


def _load_cached_models() -> list[dict] | None:
    """Return cached ``data`` if the on-disk cache exists and is fresh."""
    try:
        if not _CACHE_PATH.exists():
            return None
        if time.time() - _CACHE_PATH.stat().st_mtime > _CACHE_TTL_SECONDS:
            return None
        return json.loads(_CACHE_PATH.read_text("utf-8")).get("data")
    except Exception as exc:  # pragma: no cover - corrupt cache is non-fatal
        logger.debug("OpenRouter capability cache unreadable: %s", exc)
        return None


def _save_cached_models(models: list[dict]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps({"data": models}), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - cache write is best-effort
        logger.debug("Could not persist OpenRouter capability cache: %s", exc)


def get_capability_map(refresh: bool = False) -> dict[str, set[str]]:
    """Return the cached capability map, fetching/refreshing as needed.

    Order: in-process memo → fresh disk cache → live API. On any fetch failure
    the map degrades to ``{}`` (permissive), so dispatch is never blocked by an
    unreachable catalogue.
    """
    global _capability_map
    if _capability_map is not None and not refresh:
        return _capability_map

    models = None if refresh else _load_cached_models()
    if models is None:
        try:
            models = _fetch_models()
            _save_cached_models(models)
        except Exception as exc:
            logger.warning("OpenRouter capability fetch failed (%s); not gating schemas", exc)
            _capability_map = {}
            return _capability_map

    _capability_map = build_capability_map(models)
    return _capability_map
