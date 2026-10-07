import re
import unicodedata
from typing import Optional


class AnswerParser:

    YES_PATTERNS = {
        "có",
        "co",
        "có nhé",
        "co nhe",
        "có ạ",
        "co a",
        "có rồi",
        "co roi",
        "đúng",
        "dung",
        "đúng rồi",
        "dung roi",
        "yes",
        "có bị",
        "co bi",
    }

    NO_PATTERNS = {
        "không",
        "khong",
        "không có",
        "khong co",
        "không bị",
        "khong bi",
        "không nhé",
        "khong nhe",
        "không ạ",
        "khong a",
        "chưa",
        "chua",
        "no",
    }

    NEGATION_PATTERNS = [
        r"\bkhông\b",
        r"\bkhong\b",
        r"\bchưa\b",
        r"\bchua\b",
        r"\bkhông bị\b",
        r"\bkhong bi\b",
        r"\bkhông có\b",
        r"\bkhong co\b",
    ]

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def parse(self, text: str) -> Optional[bool]:
        normalized = self.normalize(text)

        if normalized in self.YES_PATTERNS:
            return True

        if normalized in self.NO_PATTERNS:
            return False

        return None

    def is_negated(self, text: str) -> bool:
        normalized = self.normalize(text)

        for pattern in self.NEGATION_PATTERNS:
            if re.search(pattern, normalized):
                return True

        return False

    def parse_with_context(
        self,
        text: str,
        symptom: str,
    ) -> Optional[bool]:

        normalized = self.normalize(text)

        # Explicit negative answer
        if self.is_negated(normalized):
            return False

        # Explicit positive answer
        if normalized in self.YES_PATTERNS:
            return True

        # Explicit negative answer
        if normalized in self.NO_PATTERNS:
            return False

        return None
