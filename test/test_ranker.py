from pathlib import Path

from AI.question.question_service import QuestionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models/symptom_ner_model/final"
KB_PATH = PROJECT_ROOT / "training/dataset/disease/disease_knowledge_base.csv"

def test_negative_symptom_affects_disease_ranking():

    service = QuestionService(
        model_path=str(MODEL_PATH),
        knowledge_base_path=str(KB_PATH),
    )

    state_service = service.symptom_pipeline.patient_state

    # --------------------------------
    # Tìm disease có ít nhất 2 symptoms
    # --------------------------------

    target_disease = None
    disease_symptoms = None

    for disease in service.knowledge_base.get_diseases():
        symptoms = list(
            service.knowledge_base.get_symptoms(disease)
        )

        if len(symptoms) >= 2:
            target_disease = disease
            disease_symptoms = symptoms
            break

    assert target_disease is not None
    assert disease_symptoms is not None

    positive_symptoms = disease_symptoms
    negative_symptom = disease_symptoms[0]

    print("\nTARGET DISEASE:")
    print(target_disease)

    print("\nDISEASE SYMPTOMS:")
    print(disease_symptoms)

    # --------------------------------
    # Case 1:
    # Tất cả symptoms đều TRUE
    # --------------------------------

    state_service.clear()

    for symptom in positive_symptoms:
        state_service.set_true(symptom)

    results_positive = service.rank_diseases()

    positive_item = next(
        (
            item
            for item in results_positive
            if item["disease"] == target_disease
        ),
        None,
    )

    assert positive_item is not None

    print("\n=== ALL POSITIVE ===")
    print(positive_item)

    # --------------------------------
    # Case 2:
    # Một symptom bị phủ định
    # --------------------------------

    state_service.clear()

    for symptom in positive_symptoms:
        state_service.set_true(symptom)

    state_service.set_false(
        negative_symptom
    )

    results_negative = service.rank_diseases()

    negative_item = next(
        (
            item
            for item in results_negative
            if item["disease"] == target_disease
        ),
        None,
    )

    assert negative_item is not None

    print("\n=== ONE NEGATIVE ===")
    print(negative_item)

    # --------------------------------
    # Assertions
    # --------------------------------

    assert (
        negative_symptom
        in negative_item["negative_symptoms"]
    )

    assert (
        negative_item["evidence"]["negative_penalty"]
        > 0
    )

    assert (
        negative_item["score"]
        < positive_item["score"]
    )