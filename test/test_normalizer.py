from AI.symptom.normalizer import SymptomNormalizer


normalizer = SymptomNormalizer()


tests = [
    "đau đầu",
    "nhức đầu",
    "chóng mặt",
    "hoa mắt",
    "bị sốt",
    "đau ngực",
    "khó thở",
    "buồn nôn",
]


for text in tests:
    result = normalizer.normalize(text)

    print(
        f"{text:20} -> {result}"
    )