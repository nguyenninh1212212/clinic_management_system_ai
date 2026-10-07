from AI.NER.inference import NERInference
from AI.NER.symptom_extractor import SymptomExtractor
from AI.symptom.normalizer import SymptomNormalizer
from AI.patient.state_service import PatientStateService


class SymptomPipeline:

    def __init__(self, model_path: str):

        self.ner = NERInference(model_path)

        self.extractor = SymptomExtractor()

        self.normalizer = SymptomNormalizer()

        self.patient_state = PatientStateService()

    def process(self, text: str):

        # -------------------------
        # 1. NER
        # -------------------------

        ner_result = self.ner.predict(text)

        # -------------------------
        # 2. Extract entities
        # -------------------------

        extracted = self.extractor.extract(
            ner_result["tokens"],
            ner_result["labels"],
        )

        # -------------------------
        # 3. Normalize symptoms
        # -------------------------

        normalized_symptoms = []

        for symptom in extracted["symptoms"]:

            normalized = self.normalizer.normalize(
                symptom["text"]
            )

            if normalized is None:
                continue

            normalized_symptoms.append(normalized)

            # -------------------------
            # 4. Update Patient State
            # -------------------------

            self.patient_state.update_symptom(
                code=normalized["code"],
                value=True,
            )

        return {
            "text": text,
            "entities": extracted,
            "symptoms": normalized_symptoms,
            "patient_state": self.patient_state.get_state(),
        }