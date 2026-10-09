import re
import unicodedata
from typing import Dict, List

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

    def find_in_text(self, text: str) -> List[Dict[str, str]]:
        normalized_text = self.normalize_text(text)
        matches = []

        for phrase, code in self.lookup.items():
            pattern = (
                r"(?<!\w)"
                + re.escape(phrase)
                + r"(?!\w)"
            )

            for match in re.finditer(pattern, normalized_text):
                matches.append(
                    (match.start(), match.end(), code, match.group())
                )

        matches.sort(
            key=lambda item: (item[0], -(item[1] - item[0]))
        )

        symptoms = []
        occupied_until = -1

        for start, end, code, phrase in matches:
            if start < occupied_until:
                continue

            symptoms.append({
                "code": code,
                "text": phrase,
            })
            occupied_until = end

        return symptoms