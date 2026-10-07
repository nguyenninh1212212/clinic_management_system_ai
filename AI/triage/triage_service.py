from AI.disease.disease_mapping import (
    get_specialty,
)


class TriageService:

    def triage(self, ranked_diseases):
        if not ranked_diseases:
            return None

        top_disease = ranked_diseases[0]

        disease = top_disease["disease"]

        specialty = get_specialty(disease)

        return {
            "disease": disease,
            "score": top_disease["score"],
            "specialty": specialty,
        }