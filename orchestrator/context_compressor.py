"""
ContextCompressor - Compress full conversation into fresh summary.
===================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 3, Phase D3 (Dyad-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .infrastructure.llm_client import UnifiedClient

logger = logging.getLogger(__name__)

COMPRESS_PROMPT = """Summarize this conversation into a structured context document.

Include:
1. Goal: What the user was building
2. Progress: Completed work
3. Decisions: Key architectural choices
4. Blockers: Unresolved issues
5. Next Step: Single next action

Format as Markdown. Keep under 1000 words.

Conversation:
{conversation}"""


@dataclass
class CompressedContext:
    goal: str = ""
    progress: str = ""
    decisions: str = ""
    blockers: str = ""
    next_step: str = ""
    token_estimate: int = 0
    compression_ratio: float = 0.0

    def to_markdown(self):
        nl = chr(10)
        parts = [
            "## Compressed Context",
            "",
            f"### Goal{nl}{self.goal}",
            f"### Progress{nl}{self.progress}",
            f"### Key Decisions{nl}{self.decisions}",
        ]
        if self.blockers:
            parts.append(f"### Blockers{nl}{self.blockers}")
        parts.append(f"### Next Step{nl}{self.next_step}")
        return (nl + nl).join(parts)

    @property
    def is_complete(self):
        return bool(self.goal and self.progress and self.next_step)


class ContextCompressor:
    """Compresses full conversation into a structured context."""

    def __init__(self, client=None, enabled: bool = True):
        self._client = client
        self._enabled = enabled

    def estimate_tokens(self, text):
        return len(text) // 4

    async def compress(self, conversation, max_output_tokens=600):
        input_tokens = self.estimate_tokens(conversation)

        if not self._client:
            return self._compress_fast(conversation)

        prompt = COMPRESS_PROMPT.format(conversation=conversation[-12000:])

        try:
            response = await self._client.call(
                model=None,
                prompt=prompt,
                system="You are a precise summarizer.",
                max_tokens=max_output_tokens,
                temperature=0.3,
                timeout=60,
            )
            parsed = self._parse_sections(response.text)
            output_tokens = self.estimate_tokens(response.text)
            return CompressedContext(
                goal=parsed.get("goal", ""),
                progress=parsed.get("progress", ""),
                decisions=parsed.get("decisions", ""),
                blockers=parsed.get("blockers", ""),
                next_step=parsed.get("next_step", ""),
                token_estimate=output_tokens,
                compression_ratio=input_tokens / max(output_tokens, 1),
            )
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return self._compress_fast(conversation)

    def _compress_fast(self, conversation):
        lines = conversation.split(chr(10))
        goal = ""
        progress = ""
        for line in lines:
            low = line.lower()
            if "goal" in low or "building" in low:
                goal = line.strip("#- ")[:200]
            if "done" in low or "complete" in low or "progress" in low:
                progress = line.strip("#- ")[:200]
        last = chr(10).join(lines[-10:])[:300]
        return CompressedContext(
            goal=goal or "See history",
            progress=progress or "See history",
            next_step=last,
            token_estimate=self.estimate_tokens(last),
            compression_ratio=0,
        )

    @staticmethod
    def _parse_sections(text):
        sections = {}
        current = None
        for line in text.split(chr(10)):
            line = line.strip()
            low = line.lower()
            if low.startswith("## goal"):
                current = "goal"
                sections[current] = line.split("##", 1)[-1].strip("#:- ")
            elif low.startswith("## progress"):
                current = "progress"
                sections[current] = line.split("##", 1)[-1].strip("#:- ")
            elif low.startswith("## decisions"):
                current = "decisions"
                sections[current] = line.split("##", 1)[-1].strip("#:- ")
            elif low.startswith("## blockers"):
                current = "blockers"
                sections[current] = line.split("##", 1)[-1].strip("#:- ")
            elif low.startswith("## next"):
                current = "next_step"
                sections[current] = line.split("##", 1)[-1].strip("#:- ")
            elif current and line and not line.startswith("#"):
                sections[current] = sections.get(current, "") + " " + line
        return sections

    def merge_contexts(self, *contexts):
        if not contexts:
            return CompressedContext()
        return CompressedContext(
            goal="; ".join(c.goal for c in contexts if c.goal),
            progress=chr(10).join(f"- {c.progress}" for c in contexts if c.progress),
            decisions=chr(10).join(f"- {c.decisions}" for c in contexts if c.decisions),
            blockers=chr(10).join(f"- {c.blockers}" for c in contexts if c.blockers),
            next_step=contexts[-1].next_step if contexts else "",
            token_estimate=sum(c.token_estimate for c in contexts),
            compression_ratio=0,
        )
