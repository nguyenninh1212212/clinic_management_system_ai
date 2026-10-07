from typing import List, Dict

from AI.patient.state import PatientState
from AI.disease.knowledge_base import DiseaseKnowledgeBase
from AI.disease.feature_builder import DiseaseFeatureBuilder


class DiseaseRanker:

    def __init__(
        self,
        knowledge_base: DiseaseKnowledgeBase
    ):
        self.knowledge_base = knowledge_base
        self.feature_builder = DiseaseFeatureBuilder()

    def rank(
        self,
        state: PatientState,
        top_k: int = 5
    ) -> List[Dict]:

        patient_features = (
            self.feature_builder.build(state)
        )

        patient_symptoms = set(
            patient_features.keys()
        )

        if not patient_symptoms:
            return []

        results = []

        for disease in self.knowledge_base.get_diseases():

            disease_symptoms = set(
                self.knowledge_base.get_symptoms(
                    disease
                )
            )

            matched = (
                patient_symptoms
                & disease_symptoms
            )

            if not matched:
                continue

            patient_coverage = (
                len(matched)
                / len(patient_symptoms)
            )

            disease_coverage = (
                len(matched)
                / len(disease_symptoms)
            )

            score = (
                0.7 * patient_coverage
                + 0.3 * disease_coverage
            )

            results.append({
                "disease": disease,
                "score": round(score, 4),
                "patient_coverage": round(
                    patient_coverage,
                    4
                ),
                "disease_coverage": round(
                    disease_coverage,
                    4
                ),
                "matched_symptoms": sorted(
                    matched
                ),
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]