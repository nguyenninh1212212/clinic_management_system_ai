from AI.question.question_service import QuestionService
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (PROJECT_ROOT / "models/symptom_ner_model/final")
KB_PATH = (PROJECT_ROOT/ "training/dataset/disease/disease_knowledge_base.csv")


def test_question_conversation():

    service = QuestionService(
        model_path=MODEL_PATH,
        knowledge_base_path=KB_PATH,
    )

    result = service.process_message(
        "Tôi bị đau đầu"
    )

    print("\n===== TURN 1 =====")
    print("RESULT:", result)
    print("QUESTION:", result["question"])
    print("STATE:", result["patient_state"])
    print("DISEASES:", result["ranked_diseases"])

    assert "patient_state" in result
    assert "ranked_diseases" in result

    # Answer current question
    if result["next_question"]:

        result = service.process_message("Có")

        print("\n===== TURN 2 =====")
        print("RESULT:", result)
        print("QUESTION:", result["question"])
        print("STATE:", result["patient_state"])
        print("DISEASES:", result["ranked_diseases"])

        assert "patient_state" in result