from app.schemas.patientState import PatientState


class LLMService:

    async def extract_patient_info(
        self,
        message: str,
        patient_state: PatientState,
    ) -> dict:
        pass

    async def generate_question(
        self,
        patient_state: PatientState,
    ) -> str:
        pass