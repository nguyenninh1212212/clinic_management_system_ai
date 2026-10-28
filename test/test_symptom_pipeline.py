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


def test_negation_filters_ner_symptom_and_dictionary_fallback():
    class NegatedNer:
        def predict(self, text):
            return {
                "tokens": ["<s>", "Tôi", "không", "bị", "đau", "đầu", "</s>"],
                "labels": ["B-DISEASESYMTOM", "0", "B-NEGATION", "0",
                           "B-DISEASESYMTOM", "I-DISEASESYMTOM", "B-DISEASESYMTOM"],
            }

    pipeline = make_pipeline()
    pipeline.ner = NegatedNer()
    result = pipeline.process("Tôi không bị đau đầu")
    assert result["symptoms"] == []
    assert pipeline.patient_state.get_symptom("headache").value is False


def test_negation_handles_multiple_symptoms_and_clause_boundaries():
    pipeline = make_pipeline()
    result = pipeline.process("Tôi không đau đầu và sốt nhưng bị ho")
    assert "headache" not in {s["code"] for s in result["symptoms"]}
    assert "fever" not in {s["code"] for s in result["symptoms"]}
    assert pipeline.patient_state.get_symptom("cough").value is True
    result = pipeline.process("Tôi không sốt. Tôi bị đau đầu")
    assert "headache" in {s["code"] for s in result["symptoms"]}


def test_negation_can_correct_previous_positive_state():
    pipeline = make_pipeline()
    pipeline.process("Tôi bị đau đầu")
    assert pipeline.patient_state.get_symptom("headache").value is True
    result = pipeline.process("Tôi không bị đau đầu")
    assert result["symptoms"] == []
    assert pipeline.patient_state.get_symptom("headache").value is False


def test_negation_does_not_mutate_state_when_disabled():
    pipeline = make_pipeline()
    result = pipeline.process("Tôi chẳng bị sốt", update_state=False)
    assert result["symptoms"] == []
    assert result["negated_symptoms"]
    assert pipeline.patient_state.get_state().symptoms == {}


def test_repeated_mentions_and_explicit_affirmation():
    pipeline = make_pipeline()
    result = pipeline.process("Tôi không sốt, có đau đầu")
    assert "headache" in {s["code"] for s in result["symptoms"]}
    result = pipeline.process("Trước bị đau đầu nhưng giờ không đau đầu")
    assert result["symptoms"] == []
    result = pipeline.process("Tôi không chỉ đau đầu mà còn sốt")
    assert {"headache", "fever"} <= {s["code"] for s in result["symptoms"]}


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


def test_pipeline_recognizes_dull_leg_pain_descriptions():
    pipeline = make_pipeline()

    result = pipeline.process(
        "Tôi bị đau âm ỉ ở chân"
    )

    assert [symptom["code"] for symptom in result["symptoms"]] == [
        "leg_pain",
    ]


def test_pipeline_recognizes_common_typo_in_leg_pain_description():
    pipeline = make_pipeline()

    result = pipeline.process(
        "Tôi bị đau âm ỷ ở chân"
    )

    assert [symptom["code"] for symptom in result["symptoms"]] == [
        "leg_pain",
    ]
