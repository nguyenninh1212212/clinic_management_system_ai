from typing import Optional

from .state import PatientState, SymptomState


class PatientStateService:

    def __init__(self):
        self.state = PatientState()

    def update_symptom(
        self,
        code: str,
        value: bool,
        duration: Optional[str] = None,
        severity: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:

        symptom = self.state.symptoms.get(
            code,
            SymptomState(),
        )

        symptom.value = value

        if duration is not None:
            symptom.duration = duration

        if severity is not None:
            symptom.severity = severity

        if location is not None:
            symptom.location = location

        self.state.symptoms[code] = symptom

    def get_state(self) -> PatientState:
        return self.state

    def clear(self) -> None:
        self.state = PatientState()