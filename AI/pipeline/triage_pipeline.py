from AI.NER.inference import NERInference
from AI.NER.symptom_extractor import SymptomExtractor

from AI.symptom.normalizer import SymptomNormalizer

from AI.patient.state_service import PatientStateService

from AI.disease.model import DiseaseModel
from AI.disease.ranker import DiseaseRanker

from AI.triage.triage_service import TriageService


class TriagePipeline:

    def __init__(
        self,
        ner_model_path: str,
        disease_model_path: str,
    ):
        self.ner = NERInference(ner_model_path)

        self.extractor = SymptomExtractor()
        self.normalizer = SymptomNormalizer()

        self.patient_state = PatientStateService()

        self.disease_model = DiseaseModel(
            disease_model_path
        )

        self.disease_ranker = DiseaseRanker(
            self.disease_model
        )

        self.triage_service = TriageService()

    def process(self, text: str):

        # 1. NER
        ner_result = self.ner.predict(text)

        # 2. Extract
        extracted = self.extractor.extract(
            ner_result["tokens"],
            ner_result["labels"],
        )

        # 3. Normalize
        normalized_symptoms = []

        for symptom in extracted["symptoms"]:

            normalized = self.normalizer.normalize(
                symptom["text"]
            )

            if normalized is None:
                continue

            normalized_symptoms.append(
                normalized
            )

            # 4. Patient State
            self.patient_state.update_symptom(
                code=normalized["code"],
                value=True,
            )

        # 5. Disease Ranking
        ranked_diseases = self.disease_ranker.rank(
            self.patient_state.get_state(),
            top_k=5,
        )

        # 6. Specialty
        triage = self.triage_service.triage(
            ranked_diseases
        )

        return {
            "text": text,
            "entities": extracted,
            "symptoms": normalized_symptoms,
            "patient_state": self.patient_state.get_state(),
            "diseases": ranked_diseases,
            "triage": triage,
        }