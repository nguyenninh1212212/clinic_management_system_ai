from pathlib import Path

from AI.question.question_service import QuestionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models/symptom_ner_model/final"
KB_PATH = PROJECT_ROOT / "training/dataset/disease/disease_knowledge_base.csv"


class SmallKnowledgeBase:
    diseases = ("disease_alpha", "disease_beta")
    symptoms = {
        "disease_alpha": {"headache", "cough"},
        "disease_beta": {"headache", "fever"},
    }
    weights = {
        "headache": 2.0,
        "cough": 1.0,
        "fever": 1.0,
    }

    def get_diseases(self):
        return self.diseases

    def get_symptoms(self, disease):
        return self.symptoms[disease]

    def get_symptom_weight(self, symptom):
        return self.weights[symptom]


def make_service() -> QuestionService:
    service = QuestionService(
        model_path=str(MODEL_PATH),
        knowledge_base_path=str(KB_PATH),
    )
    service.knowledge_base = SmallKnowledgeBase()
    return service


def test_ranking_score_and_missing_symptoms_for_partial_positive_evidence():
    service = make_service()
    service.symptom_pipeline.patient_state.set_true("headache")

    results = service.rank_diseases()

    assert [item["disease"] for item in results] == [
        "disease_alpha",
        "disease_beta",
    ]
    assert results[0]["score"] == 0.8833
    assert results[0]["matched_symptoms"] == ["headache"]
    assert results[0]["negative_symptoms"] == []
    assert results[0]["missing_symptoms"] == ["cough"]
    assert results[1]["missing_symptoms"] == ["fever"]


def test_explicit_negative_evidence_lowers_only_matching_disease_score():
    service = make_service()
    state = service.symptom_pipeline.patient_state
    state.set_true("headache")
    state.set_false("cough")

    results = service.rank_diseases()
    by_disease = {item["disease"]: item for item in results}

    alpha = by_disease["disease_alpha"]
    beta = by_disease["disease_beta"]

    assert alpha["negative_symptoms"] == ["cough"]
    assert alpha["evidence"]["negative_penalty"] == 0.3333
    assert alpha["score"] == 0.7833
    assert beta["score"] == 0.8833
    assert alpha["score"] < beta["score"]


def test_unknown_symptoms_do_not_count_as_positive_or_negative_evidence():
    service = make_service()
    service.symptom_pipeline.patient_state.update_symptom(
        code="headache",
        value=None,
    )

    assert service.rank_diseases() == []


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