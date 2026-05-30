"""
I18nManager - Internationalization with key extraction and LLM translation.
============================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 5, Phase R6 (Retool-inspired).
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

COMMON_LOCALES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ja",
    "zh",
    "ko",
    "ru",
    "ar",
    "hi",
    "nl",
    "sv",
    "pl",
    "tr",
]


@dataclass
class TranslationKey:
    key: str
    default_text: str
    description: str = ""
    file: str = ""


@dataclass
class LocaleFile:
    locale: str
    translations: dict = field(default_factory=dict)

    def get(self, key, default=""):
        return self.translations.get(key, default)


class I18nManager:
    """Extracts translatable strings and generates locale files."""

    def __init__(self, locales_dir="locales"):
        self._dir = Path(locales_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keys: list[TranslationKey] = []

    def extract_from_code(self, code, source_file=""):
        """Extract translatable strings from source code.

        Detects: _("text"), t("text"), i18n("text"), __("text"),
        gettext("text"), and f-string variants.
        """
        patterns = [
            r"""(?:_|t|i18n|__|gettext)\s*\(\s*["']([^"']+)["']""",
        ]
        found = []
        for pattern in patterns:
            for match in re.finditer(pattern, code):
                text = match.group(1)
                key = self._text_to_key(text)
                exists = any(k.key == key for k in self._keys)
                if not exists and len(text) > 2:
                    found.append(TranslationKey(key=key, default_text=text, file=source_file))
        self._keys.extend(found)
        return found

    @staticmethod
    def _text_to_key(text):
        key = re.sub(r"[^a-zA-Z0-9_]", "_", text[:40]).strip("_").lower()
        return key or "key_" + str(hash(text) % 10000)

    def generate_base(self, locale="en"):
        """Generate base locale file from extracted keys."""
        lf = LocaleFile(locale=locale, translations={k.key: k.default_text for k in self._keys})
        self._save_locale(lf)
        return lf

    async def translate_to(self, target_locale, client=None):
        """Translate all keys to a target locale using LLM."""
        if not client or not self._keys:
            return LocaleFile(locale=target_locale)

        base = {k.key: k.default_text for k in self._keys}
        prompt = (
            f"Translate these UI strings to {target_locale}. Return JSON: {{{{'key': 'translation', ...}}}}\n\n"
            + json.dumps(base)
        )
        try:
            response = await client.call(
                model=None,
                prompt=prompt,
                system="You are a translator.",
                max_tokens=2000,
                temperature=0.2,
                timeout=30,
            )
            translations = self._parse_translation(response.text)
            lf = LocaleFile(locale=target_locale, translations=translations)
            self._save_locale(lf)
            return lf
        except Exception:
            return LocaleFile(locale=target_locale)

    def _parse_translation(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def _save_locale(self, lf):
        (self._dir / lf.locale).mkdir(parents=True, exist_ok=True)
        fp = self._dir / lf.locale / "messages.json"
        fp.write_text(json.dumps(lf.translations, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_locale(self, locale):
        fp = self._dir / locale / "messages.json"
        if fp.exists():
            return LocaleFile(
                locale=locale, translations=json.loads(fp.read_text(encoding="utf-8"))
            )
        return LocaleFile(locale=locale)
