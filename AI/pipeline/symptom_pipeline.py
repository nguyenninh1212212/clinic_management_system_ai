from pathlib import Path
from typing import Any, Protocol

from AI.NER.inference import NERInference
from AI.NER.symptom_extractor import SymptomExtractor
from AI.symptom.normalizer import SymptomNormalizer
from AI.patient.state_service import PatientStateService


class NERPredictor(Protocol):
    def predict(self, text: str) -> dict[str, Any]: ...


class SymptomPipeline:

    def __init__(
        self,
        model_path: str | Path,
        ner: NERPredictor | None = None,
    ):

        self.ner = (
            ner
            if ner is not None
            else NERInference(str(model_path))
        )

        self.extractor = SymptomExtractor()

        self.normalizer = SymptomNormalizer()

        self.patient_state = PatientStateService()

    def process(self, text: str, update_state: bool = True):

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
        seen_codes = set()

        for symptom in extracted["symptoms"]:

            normalized = self.normalizer.normalize(
                symptom["text"]
            )

            if normalized is None or normalized["code"] in seen_codes:
                continue

            normalized_symptoms.append(normalized)
            seen_codes.add(normalized["code"])

            # -------------------------
            # 4. Update Patient State
            # -------------------------

            if update_state:
                self.patient_state.update_symptom(
                    code=normalized["code"],
                    value=True,
                )

        # Recover known symptom phrases that the NER model missed or
        # returned in a form the exact entity normalizer cannot match.
        for normalized in self.normalizer.find_in_text(text):
            if normalized["code"] in seen_codes:
                continue

            normalized_symptoms.append(normalized)
            seen_codes.add(normalized["code"])
            if update_state:
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