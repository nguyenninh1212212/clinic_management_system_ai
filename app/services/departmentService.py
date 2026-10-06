from app.schemas.patientState import PatientState


class DepartmentService:

    def recommend(
        self,
        state: PatientState,
        # triage_result: TriageResult,
    ) -> str:
        pass