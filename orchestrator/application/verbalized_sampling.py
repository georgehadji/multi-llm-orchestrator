"""
Verbalized Sampling — Distribution-Level Candidate Generation
===============================================================
CodeWhale Phase 0 — Reusable VS primitive.

Implements the paper "Verbalized Sampling" (arXiv:2510.01171v3):
  - One LLM call with a distribution-level prompt asks for k responses +
    verbalized probabilities in a single turn.
  - Probability = verbalized typicality (ordering hint), NOT quality.
  - The primitive never selects — callers bring their own quality signal.
  - Returns the full candidate list; no synthesis collapse inside this module.

Architecture:
    Application layer — depends only on domain.ports.LLMClient (port),
    never on infrastructure adapters. Passes the import-linter contract
    application-no-concrete-infra.

Usage:
    sampler = VerbalizedSampler(client=unified_client)
    candidates = await sampler.sample(
        prompt="Generate a sorting algorithm",
        model=Model.GPT_4O,
        cfg=VSConfig(k=5, probability_threshold=0.10),
    )
    # candidates[0].text, candidates[0].probability
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..models import Model, ProbabilityFormat, TaskType, VSConfig

if TYPE_CHECKING:
    from ..domain.ports import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VSCandidate:
    """A single candidate from a Verbalized Sampling call.

    text: The generated response text.
    probability: Verbalized typicality in [0, 1]. This is an ORDERING HINT
        from the model, NOT a calibrated quality score. Callers should use
        their own evaluator (e.g. EvaluatorService) for quality selection.
    """

    text: str
    probability: float = field(default=0.0)


# ── Probability format definitions ───────────────────────────────────────────

_FORMAT_DEFS = {
    ProbabilityFormat.EXPLICIT: (
        "the estimated probability from 0.0 to 1.0 of this response given the "
        "input prompt (relative to the full distribution)"
    ),
    ProbabilityFormat.CONFIDENCE: (
        "the normalized likelihood score between 0.0 and 1.0 that indicates how "
        "representative or typical this response is compared to the full "
        "distribution"
    ),
}

# ── The primitive ────────────────────────────────────────────────────────────


class VerbalizedSampler:
    """One distribution-level VS call producing diverse candidates.

    Port-only — depends on LLMClient port, never on infrastructure.
    Never selects among candidates (see VSCandidate docs).
    """

    def __init__(
        self,
        client: LLMClient,
        budget: Any = None,
    ) -> None:
        self._client = client
        self._budget = budget

    async def sample(
        self,
        *,
        prompt: str,
        model: Model,
        cfg: VSConfig = VSConfig(),
        system_extra: str = "",
        task_type: TaskType | None = None,
        max_tokens: int = 4096,
        timeout: int = 160,
    ) -> list[VSCandidate]:
        """Run one VS call → k candidates.

        Args:
            prompt: The user prompt to generate responses for.
            model: LLM to call.
            cfg: VS configuration (k, probability_threshold, fmt, temperature).
            system_extra: Additional system prompt text prepended before VS instructions.
            task_type: TaskType for response_schema routing (if applicable).
            max_tokens: Max tokens per candidate batch.
            timeout: Per-call timeout.

        Returns:
            List of VSCandidate, length <= cfg.k (may be shorter on parse failure).
            Never raises — returns [] on total failure with a warning.
        """
        system = self._build_system(cfg, system_extra)

        try:
            resp = await self._client.call(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                timeout=timeout,
                task_type=task_type,
                response_schema=bool(task_type),
            )
        except Exception as e:
            logger.warning("VerbalizedSampler call failed: %s", e)
            return []

        # Track cost against budget when available
        if self._budget is not None and resp is not None:
            try:
                cost = getattr(resp, "cost_usd", 0.0)
                await self._budget.charge(cost, "verbalized_sampling")
            except Exception:
                pass

        if not resp or not resp.text:
            logger.warning("VerbalizedSampler got empty response")
            return []

        return self._parse(resp.text, cfg.k)

    def _build_system(self, cfg: VSConfig, extra: str) -> str:
        """Build the distribution-level system prompt.

        Asks the model to generate k responses WITH verbalized probabilities
        in a single response. The probability definition changes based on
        ProbabilityFormat.

        When probability_threshold is set, appends the tail-sampling instruction
        (paper §Tail).
        """
        # Tail instruction
        tail = ""
        if cfg.probability_threshold is not None:
            tail = (
                f" Prefer responses from the low-probability tail of the "
                f"distribution — each candidate's verbalized typicality "
                f"should be below {cfg.probability_threshold}."
            )

        return (
            f"{extra}\nGenerate {cfg.k} candidate {'response' if cfg.k == 1 else 'responses'} to the user prompt. "
            f"Return ONLY valid JSON in this exact format: "
            f'{{"responses": [{{"text": str, "probability": float}}]}}. '
            f"For each response, 'probability' is "
            f"{_FORMAT_DEFS[cfg.fmt]}.{tail}"
        ).strip()

    def _parse(self, text: str, k: int) -> list[VSCandidate]:
        """Parse the LLM response into a list of VSCandidate.

        Robust parsing strategy:
          1. Strip ```json ... ``` fences
          2. Try json.loads() on the cleaned text
          3. On failure, try json5 (if available)
          4. On failure, try partial-array recovery (find `[`..`]` or `{`..`}`)
          5. On total failure, warn and return []

        Probabilities are clamped to [0, 1]. Missing probabilities default
        to uniform 1/k (with a warning). The overall function never raises.
        """
        # Step 1: Strip fences
        cleaned = self._strip_fences(text)

        # Step 2: Try standard JSON
        try:
            data = json.loads(cleaned)
            return self._extract_candidates(data, k)
        except json.JSONDecodeError:
            pass

        # Step 3: Try json5 (optional dependency)
        try:
            import json5  # type: ignore[import-untyped]

            data = json5.loads(cleaned)
            return self._extract_candidates(data, k)
        except ImportError:
            pass
        except Exception:
            pass

        # Step 4: Try partial-array recovery — find the outermost array or
        # object and extract it with regex
        candidates = self._partial_extract(cleaned, k)
        if candidates:
            return candidates

        # Step 5: Total failure
        logger.warning(
            "VerbalizedSampler: failed to parse response (k=%d, text_len=%d): %.200s",
            k,
            len(text),
            text,
        )
        return []

    # ── Internal parsing helpers ────────────────────────────────────────────

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences and surrounding whitespace."""
        text = text.strip()
        # Remove ```json ... ```, ``` ... ```
        text = re.sub(r"^```\w*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    @staticmethod
    def _extract_candidates(data: object, k: int) -> list[VSCandidate]:
        """Extract VSCandidate list from parsed JSON data.

        Accepts both {"responses": [...]} and bare [...] formats.
        """
        responses: list[dict] = []

        if isinstance(data, dict):
            data = data.get("responses", data)
            # Fallback: maybe the whole dict IS the response
            if isinstance(data, dict):
                data = [data]

        if isinstance(data, list):
            responses = data

        # Normalise each item
        candidates: list[VSCandidate] = []
        uniform_prob = 1.0 / max(k, 1)

        for item in responses[:k]:  # Take at most k
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")) if item.get("text") else ""
            if not text.strip():
                continue

            # Probability — default to uniform if missing
            prob = item.get("probability")
            if prob is None:
                logger.debug("VS candidate missing probability, defaulting to 1/k")
                prob = uniform_prob
            try:
                prob = float(prob)
            except (ValueError, TypeError):
                logger.debug("VS candidate has non-numeric probability: %r", prob)
                prob = uniform_prob

            # Clamp to [0, 1]
            prob = max(0.0, min(1.0, prob))

            candidates.append(VSCandidate(text=text, probability=prob))

        return candidates

    @staticmethod
    def _partial_extract(text: str, k: int) -> list[VSCandidate]:
        """Try regex-based partial extraction when JSON parsing fails.

        Attempts to find a JSON array or object via regex patterns.
        """
        # Try to find a {...} object with "responses" key
        obj_match = re.search(r'\{"responses"\s*:\s*\[', text)
        if obj_match:
            start = obj_match.start()
            result = VerbalizedSampler._try_extract_balanced(text, start, k, "{", "}")
            if result is not None:
                return result
            # Fallback: try to close and parse from start to end
            try:
                data = json.loads(text[start:] + "]}")
                return VerbalizedSampler._extract_candidates(data, k)
            except json.JSONDecodeError:
                try:
                    data = json.loads(text[start:] + "}")
                    return VerbalizedSampler._extract_candidates(data, k)
                except json.JSONDecodeError:
                    pass

        # Try to find a bare [...] array (anchor: [ followed by {)
        arr_match = re.search(r"\[\s*\{", text)
        if arr_match:
            start = arr_match.start()
            result = VerbalizedSampler._try_extract_balanced(text, start, k, "[", "]")
            if result is not None:
                return result
            try:
                data = json.loads(text[start:] + "]")
                if isinstance(data, list):
                    return VerbalizedSampler._extract_candidates({"responses": data}, k)
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def _try_extract_balanced(
        text: str, start: int, k: int, open_char: str, close_char: str
    ) -> list[VSCandidate] | None:
        """Try to find a balanced delimited section and parse it as JSON.

        Returns candidates on success, None on failure.
        """
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start : i + 1])
                        if open_char == "[":
                            if isinstance(data, list):
                                return VerbalizedSampler._extract_candidates(
                                    {"responses": data}, k
                                )
                        else:
                            return VerbalizedSampler._extract_candidates(data, k)
                    except json.JSONDecodeError:
                        return None
        return None


# ── Phase 5: Tier-aware VS variant selection ─────────────────────────────────
# Re-exported from models.py (domain layer — pure, no I/O).

from ..models import vs_variant_for  # noqa: F401 — re-export
