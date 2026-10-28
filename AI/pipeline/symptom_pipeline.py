from pathlib import Path
from typing import Any, Protocol

from AI.NER.symptom_extractor import SymptomExtractor
from AI.symptom.normalizer import SymptomNormalizer
from AI.patient.state_service import PatientStateService
from AI.symptom.polarity import symptom_polarity


class NERPredictor(Protocol):
    def predict(self, text: str) -> dict[str, Any]: ...


class SymptomPipeline:

    def __init__(
        self,
        model_path: str | Path,
        ner: NERPredictor | None = None,
    ):

        if ner is None:
            from AI.NER.inference import NERInference

            ner = NERInference(str(model_path))
        self.ner = ner

        self.extractor = SymptomExtractor()

        self.normalizer = SymptomNormalizer()

        self.patient_state = PatientStateService()

    def process(self, text: str, update_state: bool = True):
        print(
            "[SymptomPipeline] Processing text:",
            repr(text),
            f"(update_state={update_state})",
            flush=True,
        )

        # -------------------------
        # 1. NER
        # -------------------------

        ner_result = self.ner.predict(text)
        print("[SymptomPipeline] NER result:", ner_result, flush=True)
        # -------------------------
        # 2. Extract entities
        # -------------------------

        extracted = self.extractor.extract(
            ner_result["tokens"],
            ner_result["labels"],
        )
        print("[SymptomPipeline] Extracted entities:", extracted, flush=True)

        # -------------------------
        # 3. Normalize symptoms
        # -------------------------

        candidates = []
        for symptom in extracted["symptoms"]:
            normalized = self.normalizer.normalize(symptom["text"])
            if normalized is not None:
                candidates.append(normalized)
        candidates.extend(self.normalizer.find_in_text(text))

        normalized_symptoms = []
        negated_symptoms = []
        seen_codes = set()
        for normalized in candidates:
            code = normalized["code"]
            if code in seen_codes:
                continue
            seen_codes.add(code)
            phrases = [phrase for phrase, item_code in self.normalizer.lookup.items()
                       if item_code == code]
            value = symptom_polarity(text, phrases)
            if value is None:
                value = symptom_polarity(text, [normalized["text"]])
            value = value is not False
            if value:
                normalized_symptoms.append(normalized)
            else:
                negated_symptoms.append(normalized)
            if update_state:
                self.patient_state.update_symptom(code=code, value=value)

        print(
            "[SymptomPipeline] Normalized symptoms:",
            normalized_symptoms,
            flush=True,
        )

        return {
            "text": text,
            "entities": extracted,
            "symptoms": normalized_symptoms,
            "negated_symptoms": negated_symptoms,
            "patient_state": self.patient_state.get_state(),
        }
