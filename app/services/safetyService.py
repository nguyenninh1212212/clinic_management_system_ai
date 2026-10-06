from app.schemas.patientState import PatientState


class SafetyService:

    def check(self, state: PatientState) -> bool:
        pass

    def get_emergency_message(
        self,
        state: PatientState,
    ) -> str:
        pass