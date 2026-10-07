from typing import List, Dict, Optional

from AI.patient.state import PatientState
from AI.disease.knowledge_base import DiseaseKnowledgeBase


class QuestionEngine:

    def __init__(
        self,
        knowledge_base: DiseaseKnowledgeBase
    ):
        self.knowledge_base = knowledge_base

    def get_unknown_symptoms(
        self,
        state: PatientState,
        ranked_diseases: List[Dict]
    ) -> List[str]:
        """
        Lấy các triệu chứng chưa biết của bệnh nhân
        nhưng xuất hiện trong Top-K diseases.
        """

        unknown_symptoms = set()

        for result in ranked_diseases:

            disease = result["disease"]

            disease_symptoms = (
                self.knowledge_base.get_symptoms(
                    disease
                )
            )

            for symptom in disease_symptoms:

                symptom_state = (
                    state.symptoms.get(symptom)
                )

                # Chưa có thông tin về symptom
                if symptom_state is None:
                    unknown_symptoms.add(symptom)

                elif symptom_state.value is None:
                    unknown_symptoms.add(symptom)

        return sorted(unknown_symptoms)

    def calculate_symptom_score(
        self,
        symptom: str,
        ranked_diseases: List[Dict]
    ) -> float:
        """
        Tính điểm phân biệt của một symptom.

        Symptom xuất hiện ở nhiều bệnh khác nhau
        sẽ có điểm cao hơn vì có nhiều candidate liên quan.
        """

        if not ranked_diseases:
            return 0.0

        disease_count = 0

        for result in ranked_diseases:

            disease = result["disease"]

            disease_symptoms = set(
                self.knowledge_base.get_symptoms(
                    disease
                )
            )

            if symptom in disease_symptoms:
                disease_count += 1

        return disease_count / len(ranked_diseases)

    def select_next_symptom(
        self,
        state: PatientState,
        ranked_diseases: List[Dict]
    ) -> Optional[Dict]:
        """
        Chọn symptom tiếp theo cần hỏi.
        """

        if not ranked_diseases:
            return None

        unknown_symptoms = (
            self.get_unknown_symptoms(
                state,
                ranked_diseases
            )
        )

        if not unknown_symptoms:
            return None

        candidates = []

        for symptom in unknown_symptoms:

            score = self.calculate_symptom_score(
                symptom,
                ranked_diseases
            )

            candidates.append({
                "symptom": symptom,
                "score": round(score, 4),
            })

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates[0]

    def get_next_question(
        self,
        state: PatientState,
        ranked_diseases: List[Dict]
    ) -> Optional[Dict]:
        """
        Trả về thông tin cho câu hỏi tiếp theo.
        """

        selected = self.select_next_symptom(
            state,
            ranked_diseases
        )

        if selected is None:
            return None

        return {
            "symptom": selected["symptom"],
            "score": selected["score"],
        }
