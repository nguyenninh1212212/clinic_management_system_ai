from collections import defaultdict
import pandas as pd
import math


class DiseaseKnowledgeBase:

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.disease_symptoms = defaultdict(set)
        self.symptom_diseases = defaultdict(set)

        for row in self.df.itertuples(index=False):
            disease = row.disease
            symptom = row.symptom

            self.disease_symptoms[disease].add(symptom)
            self.symptom_diseases[symptom].add(disease)

        self.diseases = set(self.disease_symptoms.keys())

    def get_diseases(self) -> list[str]:
        return sorted(self.diseases)

    def get_symptoms(self, disease: str) -> set[str]:
        return self.disease_symptoms.get(disease, set())

    def get_diseases_by_symptom(self, symptom: str) -> set[str]:
        return self.symptom_diseases.get(symptom, set())

    def get_symptom_weight(self, symptom: str) -> float:
        total_diseases = len(self.diseases)

        frequency = len(
            self.symptom_diseases.get(symptom, set())
        )

        if total_diseases == 0 or frequency == 0:
            return 0.0

        return math.log(
            total_diseases / frequency
        )