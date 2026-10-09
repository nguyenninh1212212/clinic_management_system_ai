import re
import unicodedata
from typing import Optional


class AnswerParser:
    YES_PATTERNS = {
        "có", "co",
        "có nhé", "co nhe",
        "có ạ", "co a",
        "có rồi", "co roi",
        "đúng", "dung",
        "đúng rồi", "dung roi",
        "yes", "y",
        "có bị", "co bi",
        "có xuất hiện", "co xuat hien",
        "có gặp", "co gap",
    }

    NO_PATTERNS = {
        "không", "khong",
        "không có", "khong co",
        "không bị", "khong bi",
        "không nhé", "khong nhe",
        "không ạ", "khong a",
        "chưa", "chua",
        "no", "n",
        "không gặp", "khong gap",
        "không xuất hiện", "khong xuat hien",
    }

    # Tìm các cụm phủ định thông dụng.
    NEGATION_PATTERNS = [
        r"\bkhông\b",
        r"\bkhong\b",
        r"\bchưa\b",
        r"\bchua\b",
        r"\bkhông\s+có\b",
        r"\bkhong\s+co\b",
        r"\bkhông\s+bị\b",
        r"\bkhong\s+bi\b",
        r"\bkhông\s+gặp\b",
        r"\bkhong\s+gap\b",
        r"\bkhông\s+xuất\s+hiện\b",
        r"\bkhong\s+xuat\s+hien\b",
        r"\bkhông\s+mắc\b",
        r"\bkhong\s+mac\b",
    ]

    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text.lower().strip())
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def is_negated(self, text: str) -> bool:
        normalized = self.normalize(text)

        return any(
            re.search(pattern, normalized)
            for pattern in self.NEGATION_PATTERNS
        )

    def parse(self, text: str) -> Optional[bool]:
        normalized = self.normalize(text)

        if not normalized:
            return None

        if self.is_negated(normalized):
            return False

        if normalized in self.YES_PATTERNS:
            return True

        if normalized in self.NO_PATTERNS:
            return False

        # Chỉ chấp nhận câu trả lời đơn giản dạng "có", "không",
        # hoặc các biến thể ngắn. Nếu câu có thêm chi tiết triệu chứng,
        # nên để parser theo ngữ cảnh xử lý ở parse_with_context().
        if len(normalized.split()) > 5:
            return None

        return self.parse_with_context(text, symptom="")

    def parse_with_context(
        self,
        text: str,
        symptom: str,
    ) -> Optional[bool]:
        normalized = self.normalize(text)

        if not normalized:
            return None

        # Ưu tiên câu trả lời phủ định rõ ràng.
        # Ví dụ: "không, tôi không bị"
        #        "không gặp triệu chứng đó"
        if self.is_negated(normalized):
            return False

        # Câu trả lời xác nhận ngắn.
        if normalized in self.YES_PATTERNS:
            return True

        # Câu xác nhận dài, nhưng không chứa phủ định.
        positive_patterns = [
            r"^có(?:\s|$)",
            r"^co(?:\s|$)",
            r"^đúng(?:\s|$)",
            r"^dung(?:\s|$)",
            r"^có xuất hiện(?:\s|$)",
            r"^co xuat hien(?:\s|$)",
            r"^tôi có(?:\s|$)",
            r"^toi co(?:\s|$)",
            r"^có bị(?:\s|$)",
            r"^co bi(?:\s|$)",
        ]

        if any(re.search(p, normalized) for p in positive_patterns):
            return True

        return None
