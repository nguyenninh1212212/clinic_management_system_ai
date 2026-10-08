from pathlib import Path

from AI.question.question_service import QuestionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models/symptom_ner_model/final"
KB_PATH = PROJECT_ROOT / "training/dataset/disease/disease_knowledge_base.csv"
def test_free_form_updates_patient_state():

    service = QuestionService(
        model_path=str(MODEL_PATH),
        knowledge_base_path=str(KB_PATH),
    )

    service._merge_current_question_context(
        text="Tôi không chóng mặt nhưng hơi đau ngực",
        current_symptom="dizziness",
        symptom_result={
            "symptoms": ["chest_pain"],
        },
    )

    state = service.symptom_pipeline.patient_state.get_state()

    print("\nPATIENT STATE:")
    print(state.symptoms)

    assert state.symptoms["dizziness"].value is False
    assert state.symptoms["chest_pain"].value is True