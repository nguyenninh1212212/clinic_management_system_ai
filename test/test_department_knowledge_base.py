from pathlib import Path

from AI.department.knowledge_base import (
    DiseaseDepartmentKnowledgeBase,
)


CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "dataset"
    / "department"
    / "disease_department.csv"
)


def test_load_knowledge_base():
    kb = DiseaseDepartmentKnowledgeBase(
        str(CSV_PATH)
    )

    mappings = kb.get_all_mappings()

    assert len(mappings) == 627


def test_get_department():
    kb = DiseaseDepartmentKnowledgeBase(
        str(CSV_PATH)
    )

    department = kb.get_department(
        "pseudotumor_cerebri"
    )

    assert department == "Thần kinh"


def test_review_disease_returns_none():
    kb = DiseaseDepartmentKnowledgeBase(
        str(CSV_PATH)
    )

    department = kb.get_department(
        "amyloidosis"
    )

    assert department is None