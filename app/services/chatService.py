from app.schemas.patientState import PatientState
from app.services.departmentService import DepartmentService
from app.services.extractionService import ExtractionService
from app.services.questionService import QuestionService
from app.services.safetyService import SafetyService
from app.services.triageService import TriageService


class ChatService:

    def __init__(
        self,
        extraction_service: ExtractionService,
        safety_service: SafetyService,
        question_service: QuestionService,
        triage_service: TriageService,
        department_service: DepartmentService,
    ):
        self.extraction_service = extraction_service
        self.safety_service = safety_service
        self.question_service = question_service
        self.triage_service = triage_service
        self.department_service = department_service

    async def chat(
    self,
    message: str,
    state: PatientState,
):

    # 1. Extract information
        state = await self.extraction_service.extract(
        message,
        state,
    )

    # 2. Safety
        if self.safety_service.check(state):
            return {
            "status": "emergency",
            "message": self.safety_service.get_emergency_message(state),
            "patient_state": state,
            }

    # 3. Missing information
        missing = self.question_service.get_missing_information(state)

        if missing:
            question = await self.question_service.generate_next_question(state)

            return {
            "status": "collecting_information",
            "message": question,
            "patient_state": state,
        }

    # 4. Triage
        triage = self.triage_service.predict(state)

        # 5. Department
        department = self.department_service.recommend(
        state,
        triage,
    )

        return {
        "status": "completed",
        "message": f"Bạn nên khám {department}.",
        "patient_state": state,
        "triage": triage,
        "department": department,
    }