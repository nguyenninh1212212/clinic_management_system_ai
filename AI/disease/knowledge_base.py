import pandas as pd
from pathlib import Path
from typing import List, Dict


class DiseaseKnowledgeBase:

    def __init__(self, path: str):
        self.path = Path(path)

        self.df = pd.read_csv(
            self.path
        )

        self.disease_to_symptoms = (
            self.df
            .groupby("disease")["symptom"]
            .apply(list)
            .to_dict()
        )

        self.symptom_to_diseases = (
            self.df
            .groupby("symptom")["disease"]
            .apply(list)
            .to_dict()
        )

    def get_diseases(self) -> List[str]:
        return list(
            self.disease_to_symptoms.keys()
        )

    def get_symptoms(
        self,
        disease: str
    ) -> List[str]:

        return self.disease_to_symptoms.get(
            disease,
            []
        )

    def get_diseases_by_symptom(
        self,
        symptom: str
    ) -> List[str]:

        return self.symptom_to_diseases.get(
            symptom,
            []
        )

    def has_disease(
        self,
        disease: str
    ) -> bool:

        return disease in self.disease_to_symptoms

    def has_symptom(
        self,
        symptom: str
    ) -> bool:

        return symptom in self.symptom_to_diseases