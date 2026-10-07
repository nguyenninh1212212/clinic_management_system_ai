import re
import unicodedata

from .dictionary import SYMPTOM_DICTIONARY


class SymptomNormalizer:

    def __init__(self):
        self.dictionary = SYMPTOM_DICTIONARY

        self.lookup = {}

        for code, phrases in self.dictionary.items():
            for phrase in phrases:
                self.lookup[self.normalize_text(phrase)] = code

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().strip()

        text = unicodedata.normalize(
            "NFC",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    def normalize(self, text: str):
        normalized_text = self.normalize_text(text)

        code = self.lookup.get(normalized_text)

        if code is None:
            return None

        return {
            "code": code,
            "text": text,
        }