import re


class PIIMaskingETL:
    def __init__(self):
        self.rules = [
            (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "<EMAIL_MASKED>"),
            (
                re.compile(
                    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
                ),
                "<IPV4_MASKED>",
            ),
            (re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"), "<API_KEY_MASKED>"),
            (
                re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
                "<PHONE_MASKED>",
            ),
        ]

    def transform(self, text: str) -> str:
        for pattern, replacement in self.rules:
            text = pattern.sub(replacement, text)
        return text

    def process_dict(self, data: dict) -> dict:
        result = {}
        for k, v in data.items():
            if isinstance(v, str):
                result[k] = self.transform(v)
            elif isinstance(v, dict):
                result[k] = self.process_dict(v)
            elif isinstance(v, list):
                result[k] = [
                    (
                        self.transform(item)
                        if isinstance(item, str)
                        else self.process_dict(item) if isinstance(item, dict) else item
                    )
                    for item in v
                ]
            else:
                result[k] = v
        return result
