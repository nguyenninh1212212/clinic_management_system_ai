from typing import List, Dict


class DiseaseRanker:
    def __init__(self, disease_model):
        self.disease_model = disease_model

    def build_text(self, patient_state) -> str:
        features = []

        for code, symptom in patient_state.symptoms.items():
            if symptom.value is True:
                features.append(code)

            elif symptom.value is False:
                features.append(f"{code}_absent")

        return " ".join(features)

    def rank(
        self,
        patient_state,
        top_k: int = 5,
    ):
        text = self.build_text(patient_state)

        if not text:
            return []

        scores = self.disease_model.predict_proba(text)

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {
                "disease": disease,
                "score": round(score, 4),
            }
            for disease, score in ranked[:top_k]
        ]

