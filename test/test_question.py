from pathlib import Path

from AI.disease.knowledge_base import DiseaseKnowledgeBase
from AI.disease.ranker import DiseaseRanker

from AI.patient.state import (
    PatientState,
    SymptomState,
)

from AI.question.question_engine import QuestionEngine


# ============================================================
# 1. Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 2. Knowledge Base
# ============================================================

KB_PATH = (
    PROJECT_ROOT
    / "training"
    / "dataset"
    / "disease"
    / "disease_knowledge_base.csv"
)

kb = DiseaseKnowledgeBase(KB_PATH)


print("=" * 60)
print("QUESTION ENGINE TEST")
print("=" * 60)

print(
    f"\nNumber of diseases: "
    f"{len(kb.get_diseases())}"
)


# ============================================================
# 3. Patient State
# ============================================================

state = PatientState()


state.symptoms["headache"] = SymptomState(
    value=True
)

state.symptoms["dizziness"] = SymptomState(
    value=True
)

state.symptoms["nausea"] = SymptomState(
    value=True
)


print("\nPatient State:")

for code, symptom in state.symptoms.items():

    print(
        f"- {code}: {symptom.value}"
    )


# ============================================================
# 4. Disease Ranker
# ============================================================

ranker = DiseaseRanker(kb)


results = ranker.rank(
    state,
    top_k=5
)


print("\nTop-K Diseases:")

for index, result in enumerate(
    results,
    start=1
):

    print(
        f"{index}. "
        f"{result['disease']} "
        f"({result['score']})"
    )


# ============================================================
# 5. Question Engine
# ============================================================

question_engine = QuestionEngine(kb)


next_question = (
    question_engine.get_next_question(
        state,
        results
    )
)


# ============================================================
# 6. Result
# ============================================================

print("\nNext Question:")

if next_question is None:

    print("No question available.")

else:

    print(
        f"Symptom: "
        f"{next_question['symptom']}"
    )

    print(
        f"Score: "
        f"{next_question['score']}"
    )


print("\n" + "=" * 60)
print("TEST FINISHED")
print("=" * 60)
