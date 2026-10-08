from pathlib import Path

from AI.question.question_service import QuestionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models/symptom_ner_model/final"
KB_PATH = (
    PROJECT_ROOT
    / "training/dataset/disease/disease_knowledge_base.csv"
)


def test_question_selection():

    service = QuestionService(
        model_path=str(MODEL_PATH),
        knowledge_base_path=str(KB_PATH),
    )

    state_service = (
        service.symptom_pipeline.patient_state
    )

    state_service.clear()

    state_service.set_true("headache")

    ranked_diseases = service.rank_diseases()

    print("\nRANKED DISEASES:")

    for item in ranked_diseases[:10]:
        print(
            item["disease"],
            item["score"],
        )

    symptom = service.select_best_question_symptom(
        ranked_diseases
    )

    print("\nSELECTED SYMPTOM:")
    print(symptom)

    assert symptom is not None

    assert (
        symptom
        not in state_service.get_state().symptoms
    )