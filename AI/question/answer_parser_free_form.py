import re
from typing import Optional

from AI.symptom.dictionary import SYMPTOM_DICTIONARY


class FreeFormAnswerParser:

    def parse(self, text: str) -> dict:
        text = self._normalize(text)

        return {
            "duration": self._extract_duration(text),
            "severity": self._extract_severity(text),
        }

    def parse_symptom_polarity(
        self,
        text: str,
        symptoms: list[str],
        current_symptom: Optional[str] = None,
    ) -> dict:
        """
        Xác định polarity của từng symptom trong câu trả lời.

        Ví dụ:
            "Tôi không khó thở nhưng đau ngực"

        =>
            {
                "shortness_of_breath": False,
                "chest_pain": True
            }
        """

        normalized = self._normalize(text)

        result = {}

        for symptom in symptoms:
            value = self._detect_symptom_polarity(
                normalized,
                symptom,
            )

            if value is not None:
                result[symptom] = value

        # Nếu NER không tìm được current symptom nhưng
        # câu trả lời vẫn là câu trả lời trực tiếp cho câu hỏi
        if current_symptom and current_symptom not in result:
            if self._contains_negation_for_symptom(
                normalized,
                current_symptom,
            ):
                result[current_symptom] = False
            elif self._contains_positive_for_symptom(
                normalized,
                current_symptom,
            ):
                result[current_symptom] = True

        return result

    # -------------------------------------------------
    # Symptom polarity
    # -------------------------------------------------

    def _detect_symptom_polarity(
        self,
        text: str,
        symptom: str,
    ) -> Optional[bool]:

        symptom_words = self._symptom_words(symptom)

        if not symptom_words:
            return None

        symptom_position = self._find_symptom_position(
            text,
            symptom_words,
        )

        if symptom_position is None:
            return None

        # -------------------------------------------------
        # Chỉ xét phần ngay trước symptom.
        #
        # Ví dụ:
        # "tôi không khó thở nhưng đau ngực"
        #
        # với "đau ngực":
        #
        # context = "tôi không khó thở nhưng"
        #
        # Từ "nhưng" là boundary -> "không" phía trước
        # không còn tác động đến "đau ngực".
        # -------------------------------------------------

        context = text[:symptom_position]

        # Các liên từ tạo ranh giới polarity
        boundaries = [
            " nhưng ",
            " tuy nhiên ",
            " còn ",
            " mà ",
            " và ",
            ",",
            ";",
        ]

        last_boundary = -1

        for boundary in boundaries:
            position = context.rfind(boundary)

            if position > last_boundary:
                last_boundary = position

        if last_boundary >= 0:
            context = context[last_boundary + 1:]

        # -------------------------------------------------
        # Check negation trong scope gần symptom
        # -------------------------------------------------

        if self._contains_negation(context):
            return False

        return True


    def _contains_negation(
        self,
        text: str,
    ) -> bool:

        patterns = [
            r"\bkhông\b",
            r"\bchưa\b",
            r"\bchẳng\b",
            r"\bkhông hề\b",
            r"\bkhông bị\b",
            r"\bkhông có\b",
            r"\bchưa từng\b",
        ]

        return any(
            re.search(pattern, text)
            for pattern in patterns
        )
    # -------------------------------------------------
    # Symptom matching
    # -------------------------------------------------

    def _symptom_words(self, symptom: str) -> list[str]:
        """
        Chuyển symptom code thành các keyword đơn giản.

        Ví dụ:
            shortness_of_breath
            =>
            ["khó thở", "hụt hơi"]

            chest_pain
            =>
            ["đau ngực"]
        """

        phrases = SYMPTOM_DICTIONARY.get(symptom)
        if phrases:
            return phrases

        return [symptom.replace("_", " ")]

    def _find_symptom_position(
        self,
        text: str,
        symptom_words: list[str],
    ) -> Optional[int]:

        positions = [
            match.start()
            for word in symptom_words
            for match in re.finditer(
                rf"(?<!\w){re.escape(word)}(?!\w)",
                text,
            )
        ]

        if not positions:
            return None

        return min(positions)

    # -------------------------------------------------
    # Negation
    # -------------------------------------------------

    def _contains_negation(
        self,
        text: str,
    ) -> bool:

        patterns = [
            r"\bkhông\b",
            r"\bchưa\b",
            r"\bchẳng\b",
            r"\bkhông hề\b",
            r"\bkhông bị\b",
            r"\bkhông có\b",
            r"\bchưa từng\b",
        ]

        return any(
            re.search(pattern, text)
            for pattern in patterns
        )

    def _contains_negation_for_symptom(
        self,
        text: str,
        symptom: str,
    ) -> bool:

        for word in self._symptom_words(symptom):
            position = text.find(word)

            if position < 0:
                continue

            context = text[
                max(0, position - 35):position
            ]

            if self._contains_negation(context):
                return True

        return False

    def _contains_positive_for_symptom(
        self,
        text: str,
        symptom: str,
    ) -> bool:

        return self._find_symptom_position(
            text,
            self._symptom_words(symptom),
        ) is not None

    # -------------------------------------------------
    # Duration
    # -------------------------------------------------

    def _extract_duration(
        self,
        text: str,
    ) -> Optional[str]:

        patterns = [
            r"(?:từ|khoảng)\s+\d+\s+(?:ngày|tuần|tháng|giờ)",
            r"\d+\s+(?:ngày|tuần|tháng|giờ)\s+nay",
            r"\d+\s+(?:ngày|tuần|tháng|giờ)",
            r"hôm qua",
            r"hôm nay",
            r"mấy hôm",
            r"vài ngày",
            r"một thời gian",
            r"gần đây",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                return match.group(0)

        return None

    # -------------------------------------------------
    # Severity
    # -------------------------------------------------

    def _extract_severity(
        self,
        text: str,
    ) -> Optional[str]:

        severe = [
            "rất đau",
            "đau dữ dội",
            "đau dữ",
            "đau nhiều",
            "rất khó chịu",
            "nghiêm trọng",
        ]

        moderate = [
            "khá đau",
            "khá nhiều",
            "vừa phải",
            "trung bình",
        ]

        mild = [
            "hơi",
            "nhẹ",
            "một chút",
            "không nhiều",
        ]

        if any(word in text for word in severe):
            return "severe"

        if any(word in text for word in moderate):
            return "moderate"

        if any(word in text for word in mild):
            return "mild"

        return None

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()

        return re.sub(
            r"\s+",
            " ",
            text,
        )