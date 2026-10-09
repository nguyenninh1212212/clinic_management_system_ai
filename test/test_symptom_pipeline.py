from AI.NER.symptom_extractor import SymptomExtractor
from AI.patient.state_service import PatientStateService
from AI.pipeline.symptom_pipeline import SymptomPipeline
from AI.symptom.normalizer import SymptomNormalizer


class NoEntityNer:
    def predict(self, text: str) -> dict:
        return {
            "tokens": ["<s>", "</s>"],
            "labels": ["O", "O"],
        }


def make_pipeline() -> SymptomPipeline:
    pipeline = SymptomPipeline.__new__(SymptomPipeline)
    pipeline.ner = NoEntityNer()
    pipeline.extractor = SymptomExtractor()
    pipeline.normalizer = SymptomNormalizer()
    pipeline.patient_state = PatientStateService()
    return pipeline


def test_pipeline_recovers_symptoms_from_description_when_ner_misses_them():
    pipeline = make_pipeline()

    result = pipeline.process(
        "Tôi cảm thấy tay chân yếu hơn bình thường "
        "và đôi lúc khó giữ thăng bằng."
    )

    assert {symptom["code"] for symptom in result["symptoms"]} == {
        "weakness",
        "dizziness",
    }


def test_pipeline_normalizes_symptom_phrases_outside_ner_entities():
    pipeline = make_pipeline()

    result = pipeline.process(
        "Tôi bị đau dầu tê tay và sổ mũi."
    )

    assert {symptom["code"] for symptom in result["symptoms"]} == {
        "headache",
        "paresthesia",
        "coryza",
    }
