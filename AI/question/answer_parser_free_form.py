import re
import unicodedata
from typing import Optional


class FreeFormAnswerParser:

    DURATION_PATTERNS = [
        r"(\d+)\s*(ngày|ngay)",
        r"(\d+)\s*(giờ|gio)",
        r"(\d+)\s*(tuần|tuan)",
        r"(\d+)\s*(tháng|thang)",
    ]

    SEVERITY_MAP = {
        "nhẹ": "mild",
        "nhe": "mild",

        "vừa": "moderate",
        "vua": "moderate",
        "trung bình": "moderate",
        "trung binh": "moderate",

        "nặng": "severe",
        "nang": "severe",
        "rất nặng": "severe",
        "rat nang": "severe",
    }

    @staticmethod
    def normalize(text: str) -> str:
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

    def extract_duration(
        self,
        text: str,
    ) -> Optional[str]:

        normalized = self.normalize(text)

        # "3 ngày"
        for pattern in self.DURATION_PATTERNS:

            match = re.search(
                pattern,
                normalized,
            )

            if match:
                number = match.group(1)
                unit = match.group(2)

                return f"{number} {unit}"

        # "từ hôm qua"
        if "từ hôm qua" in normalized:
            return "từ hôm qua"

        if "tu hom qua" in normalized:
            return "từ hôm qua"

        # "từ sáng"
        if "từ sáng" in normalized:
            return "từ sáng"

        if "tu sang" in normalized:
            return "từ sáng"

        # "từ tối qua"
        if "từ tối qua" in normalized:
            return "từ tối qua"

        return None

    def extract_severity(
        self,
        text: str,
    ) -> Optional[str]:

        normalized = self.normalize(text)

        # Check longest phrases first
        severity_items = sorted(
            self.SEVERITY_MAP.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

        for phrase, severity in severity_items:

            if phrase in normalized:
                return severity

        return None

    def parse(
        self,
        text: str,
    ) -> dict:

        return {
            "duration": self.extract_duration(text),
            "severity": self.extract_severity(text),
        }