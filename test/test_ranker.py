from pathlib import Path

from AI.disease.knowledge_base import DiseaseKnowledgeBase
from AI.disease.ranker import DiseaseRanker
from AI.patient.state import PatientState, SymptomState


print("=" * 60)
print("DISEASE RANKER TEST")
print("=" * 60)


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KB_PATH = (
    PROJECT_ROOT
    / "training"
    / "dataset"
    / "disease"
    / "disease_knowledge_base.csv"
)

print("\n[1] Project root:")
print(PROJECT_ROOT)

print("\n[2] Knowledge Base path:")
print(KB_PATH)

print("\n[3] Knowledge Base exists:")
print(KB_PATH.exists())


if not KB_PATH.exists():
    raise FileNotFoundError(
        f"Knowledge Base not found:\n{KB_PATH}"
    )


# ============================================================
# 2. Load Knowledge Base
# ============================================================

print("\n[4] Loading Knowledge Base...")

kb = DiseaseKnowledgeBase(KB_PATH)

print("Knowledge Base loaded successfully.")

diseases = kb.get_diseases()

print(f"Number of diseases: {len(diseases)}")


# ============================================================
# 3. Test Knowledge Base
# ============================================================

print("\n[5] Test Knowledge Base")

test_disease = diseases[0]

print(f"Test disease: {test_disease}")

disease_symptoms = kb.get_symptoms(test_disease)

print("Symptoms:")

for symptom in disease_symptoms[:10]:
    print(f"  - {symptom}")

print(
    f"Total symptoms: {len(disease_symptoms)}"
)


