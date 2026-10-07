from typing import Optional

from AI.patient.state import (
    PatientState,
    SymptomState,
)


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
    ) -> None:

        symptom = self.state.symptoms.get(
            code,
            SymptomState(),
        )

        if value is not None:
            symptom.value = value

        if duration is not None:
            symptom.duration = duration

        if severity is not None:
            symptom.severity = severity

        if location is not None:
            symptom.location = location

        self.state.symptoms[code] = symptom

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