from typing import Optional

from AI.patient.state import (
    PatientState,
    SymptomState,
)

VALID_ASSERTIONS = {
        "AFFIRMED",
        "NEGATED",
        "ABSENT_NOW",
        "UNCERTAIN",
    }
class PatientStateService:

    def __init__(self):
        self.state = PatientState()

        



    def update_symptom(
        self,
        code: str,
        value: Optional[bool] = None,
        duration: Optional[str] = None,
        severity: Optional[str] = None,
        location: Optional[str] = None,
        assertion: Optional[str] = None,
    ) -> None:
        """
        Cập nhật trạng thái triệu chứng.

        Tương thích với code cũ:
            update_symptom(code="headache", value=True)

        Cách gọi mới:
            update_symptom(code="headache", assertion="NEGATED")
        """

        if assertion is not None and assertion not in VALID_ASSERTIONS:
            raise ValueError(
                f"Invalid assertion: {assertion}. "
                f"Expected one of {sorted(VALID_ASSERTIONS)}"
            )

        if assertion is not None:
            assertion_to_value = {
                "AFFIRMED": True,
                "NEGATED": False,
                "ABSENT_NOW": False,
                "UNCERTAIN": None,
            }
            value = assertion_to_value[assertion]
        elif value is not None:
            assertion = "AFFIRMED" if value else "NEGATED"

        symptom = self.state.symptoms.get(code)

        if symptom is None:
            symptom = SymptomState()
            self.state.symptoms[code] = symptom

        # Chỉ cập nhật value/assertion khi có thông tin mới.
        # value=None từ code cũ không tự động xóa giá trị đã lưu.
        if assertion is not None:
            symptom.assertion = assertion
            symptom.value = value
        elif value is not None:
            symptom.value = value
            symptom.assertion = "AFFIRMED" if value else "NEGATED"

        if duration is not None:
            symptom.duration = duration

        if severity is not None:
            symptom.severity = severity

        if location is not None:
            symptom.location = location
    def set_true(
        self,
        code: str,
        duration: Optional[str] = None,
        severity: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:

        self.update_symptom(
            code=code,
            value=True,
            duration=duration,
            severity=severity,
            location=location,
        )

    def set_false(self, code: str) -> None:
        self.update_symptom(
            code=code,
            value=False,
        )

    def set_unknown(self, code: str) -> None:
        self.update_symptom(
            code=code,
            value=None,
        )

    def get_symptom(
        self,
        code: str,
    ) -> Optional[SymptomState]:

        return self.state.symptoms.get(code)

    def get_state(self) -> PatientState:
        return self.state

    def clear(self) -> None:
        self.state = PatientState()