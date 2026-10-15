from AI.symptom.normalizer import SymptomNormalizer


def test_normalizer_supports_string_dictionary_values():
    normalizer = SymptomNormalizer()

    assert normalizer.normalize("dị ứng") == {
        "code": "allergic_reaction",
        "text": "dị ứng",
    }


def test_normalizer_splits_pipe_separated_dictionary_values():
    normalizer = SymptomNormalizer()

    assert normalizer.normalize("ngứa da") == {
        "code": "itching_of_skin",
        "text": "ngứa da",
    }
    assert normalizer.normalize("ngứa người") == {
        "code": "itching_of_skin",
        "text": "ngứa người",
    }
