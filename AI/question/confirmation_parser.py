import re


class ConfirmationParser:

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def parse(self, text: str) -> dict:

        original_text = text.strip()
        normalized_text = original_text.lower()

        confirmations = (
            self.knowledge_base
            .get_confirmations()
        )

        for item in confirmations:
            phrase = str(
                item["text"]
            ).strip().lower()

            if not phrase:
                continue

            pattern = (
                rf"(?<!\w)"
                rf"{re.escape(phrase)}"
                rf"(?!\w)"
            )

            match = re.search(
                pattern,
                normalized_text,
            )

            if not match:
                continue

            remaining_text = (
                original_text[:match.start()]
                + original_text[match.end():]
            ).strip()

            return {
                "type": "confirmation",
                "value": (
                    item["confirmation_type"]
                    == "yes"
                ),
                "confidence": "high",
                "matched_text": (
                    match.group(0)
                ),
                "remaining_text": remaining_text,
            }

        return {
            "type": "confirmation",
            "value": None,
            "confidence": "none",
            "matched_text": None,
            "remaining_text": original_text,
        }