import re
import unicodedata


class SafetyTriage:
    EMERGENCY_MESSAGE = (
        "Các dấu hiệu bạn mô tả có thể cần được đánh giá khẩn cấp. "
        "Vui lòng gọi cấp cứu địa phương hoặc đến khoa cấp cứu ngay; "
        "không chờ chatbot tiếp tục sàng lọc. Nếu có thể, nhờ người "
        "ở cạnh hỗ trợ."
    )

    NEGATION_PATTERN = re.compile(
        r"\b(?:khong|chua|chang|khong bi|khong co|khong gap|"
        r"khong xuat hien|chua tung)\b"
    )
    CLAUSE_BOUNDARY_PATTERN = re.compile(
        r"[,.!?;]|\bnhung\b|\btuy nhien\b|\bcon\b"
    )
    SENTENCE_BOUNDARY_PATTERN = re.compile(
        r"[.!?;]|\bnhung\b|\btuy nhien\b|\bcon\b"
    )

    def assess(self, text: str) -> dict | None:
        normalized = self._normalize(text)
        red_flags = []

        if self._has_any(
            normalized,
            (
                "khong tho duoc",
                "khong the tho",
                "khong tho noi",
                "tho khong duoc",
                "kho tho du doi",
                "kho tho nghiem trong",
                "kho tho cap tinh",
                "rat kho tho",
                "kho tho nhieu",
                "tim tai",
            ),
        ):
            red_flags.append("severe_breathing_difficulty")

        chest_pain = self._matches(
            normalized,
            (
                "dau nguc",
                "tuc nguc",
                "bop nghe nguc",
                "nang nguc",
            ),
        )
        severe_cues = self._matches(
            normalized,
            (
                "du doi",
                "nghiem trong",
                "rat nang",
                "nang",
                "dau nhieu",
                "nhieu",
                "dot ngot",
                "vo mo hoi",
                "ngat",
                "kho tho",
            ),
        )
        if any(
            self._same_sentence(normalized, pain.start(), cue.start())
            and not self._is_negated(normalized, pain.start())
            and not self._is_negated(normalized, cue.start())
            for pain in chest_pain
            for cue in severe_cues
        ):
            red_flags.append("severe_or_complicated_chest_pain")

        stroke_signs = self._matches(
            normalized,
            (
                "meo mieng",
                "noi kho",
                "noi khong ro",
                "yeu liet mot ben",
                "te liet mot ben",
                "yeu mot ben",
                "te mot ben",
            ),
        )
        sudden_weakness = self._matches(
            normalized,
            (
                "dot ngot yeu",
                "yeu dot ngot",
                "dot ngot te",
                "dot ngot bi liet",
            ),
        )
        if any(
            not self._is_negated(normalized, match.start())
            for match in (*stroke_signs, *sudden_weakness)
        ):
            red_flags.append("possible_sudden_neurological_deficit")

        if self._has_any(
            normalized,
            (
                "bat tinh",
                "ngat",
                "co giat",
                "chay mau khong cam",
            ),
        ):
            red_flags.append("loss_of_consciousness_seizure_or_severe_bleeding")

        if not red_flags:
            return None

        return {
            "urgency": "HIGH",
            "red_flags": red_flags,
            "message": self.EMERGENCY_MESSAGE,
        }

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text.lower())
        normalized = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        normalized = normalized.replace("đ", "d")
        normalized = re.sub(r"[^\w\s,.!?;]", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _matches(text: str, phrases: tuple[str, ...]) -> list[re.Match[str]]:
        return [
            match
            for phrase in phrases
            for match in re.finditer(
                rf"(?<!\w){re.escape(phrase)}(?!\w)",
                text,
            )
        ]

    def _has_any(self, text: str, phrases: tuple[str, ...]) -> bool:
        return any(
            not self._is_negated(text, match.start())
            for match in self._matches(text, phrases)
        )

    def _is_negated(self, text: str, position: int) -> bool:
        prefix = text[:position]
        boundaries = list(self.CLAUSE_BOUNDARY_PATTERN.finditer(prefix))
        if boundaries:
            prefix = prefix[boundaries[-1].end():]
        return self.NEGATION_PATTERN.search(prefix) is not None

    def _same_sentence(
        self,
        text: str,
        first_position: int,
        second_position: int,
    ) -> bool:
        start = min(first_position, second_position)
        end = max(first_position, second_position)
        between = text[start:end]
        if self.SENTENCE_BOUNDARY_PATTERN.search(between):
            return False
        return True
