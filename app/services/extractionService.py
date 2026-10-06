from app.schemas.patientState import PatientState


class ExtractionService:

    async def extract(
        self,
        message: str,
        patient_state: PatientState,
    ) -> PatientState:

        # LLM/NLP
        # message → structured information

        return patient_state