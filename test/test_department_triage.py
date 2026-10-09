from pathlib import Path

from AI.department.knowledge_base import (
    DiseaseDepartmentKnowledgeBase,
)

from AI.triage.triage_service import (
    DepartmentTriageService,
)


CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "dataset"
    / "department"
    / "disease_department.csv"
)


def create_service():
    kb = DiseaseDepartmentKnowledgeBase(
        str(CSV_PATH)
    )

    return DepartmentTriageService(kb)


def test_suggest_department():
    service = create_service()

    ranked_diseases = [
        {
            "disease": "pseudotumor_cerebri",
            "score": 0.86,
        },
        {
            "disease": "meniere_disease",
            "score": 0.72,
        },
        {
            "disease": "atelectasis",
            "score": 0.64,
        },
    ]

    result = service.suggest_department(
        ranked_diseases
    )

    assert result["suggested_department"] == "Thần kinh"
    assert result["score"] > 0

    departments = result["departments"]

    assert len(departments) >= 2


def test_empty_candidates():
    service = create_service()

    result = service.suggest_department([])

    assert result["suggested_department"] is None
    assert result["score"] == 0.0
    assert result["departments"] == []
    assert result["status"] == "insufficient_evidence"


def test_review_disease_is_ignored():
    service = create_service()

    ranked_diseases = [
        {
            "disease": "amyloidosis",
            "score": 0.95,
        },
        {
            "disease": "pseudotumor_cerebri",
            "score": 0.80,
        },
    ]

    result = service.suggest_department(
        ranked_diseases
    )

    assert (
        result["suggested_department"]
        == "Thần kinh"
    )


def test_candidates_with_only_unreviewed_mappings_have_no_suggestion():
    service = create_service()

    result = service.suggest_department(
        [
            {"disease": "amyloidosis", "score": 0.95},
            {"disease": "unknown_disease", "score": 0.90},
        ]
    )

    assert result["suggested_department"] is None
    assert result["departments"] == []
    assert result["status"] == "insufficient_evidence"


def test_equal_department_scores_are_reported_as_ambiguous():
    service = create_service()
    departments = {
        service.knowledge_base.get_department(disease)
        for disease in ("pseudotumor_cerebri", "atelectasis")
    }
    assert len(departments) == 2

    result = service.suggest_department(
        [
            {"disease": "pseudotumor_cerebri", "score": 0.8},
            {"disease": "atelectasis", "score": 0.8},
        ]
    )

    assert result["suggested_department"] is None
    assert result["status"] == "ambiguous"
    assert len(result["departments"]) == 2


def test_print_department_triage_result():
    service = create_service()

    ranked_diseases = [
        {
            "disease": "pseudotumor_cerebri",
            "score": 0.86,
        },
        {
            "disease": "meniere_disease",
            "score": 0.72,
        },
        {
            "disease": "atelectasis",
            "score": 0.64,
        },
        {
            "disease": "meningioma",
            "score": 0.60,
        },
    ]

    result = service.suggest_department(
        ranked_diseases
    )

    print("\n" + "=" * 60)
    print("DEPARTMENT TRIAGE RESULT")
    print("=" * 60)

    print(
        f"Suggested department: "
        f"{result['suggested_department']}"
    )

    print(
        f"Score: "
        f"{result['score']}"
    )

    print("\nDepartments:")

    for item in result["departments"]:
        print(
            f"  - {item['department']}: "
            f"{item['score']}"
        )

        print(
            f"    diseases: "
            f"{', '.join(item['diseases'])}"
        )

    print("=" * 60)

    assert result["suggested_department"] == "Thần kinh"