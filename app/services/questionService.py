from app.schemas.patientState import PatientState


class QuestionService:

    def get_missing_information(
        self,
        state: PatientState,
    ) -> list[str]:
        pass

    async def generate_next_question(
        self,
        state: PatientState,
    ) -> str:
        pass