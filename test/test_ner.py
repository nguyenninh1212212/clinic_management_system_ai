from AI.NER.inference import NERInference
from AI.NER.symptom_extractor import SymptomExtractor
MODEL_PATH = (
    r"D:\project\clinic_management_system_ai"
    r"\models\symptom_ner_model\final"
)
inference = NERInference(MODEL_PATH)
extractor = SymptomExtractor()


text = "Tôi bị đau đầu và chóng mặt trong 3 ngày nay"


# 1. PhoBERT
result = inference.predict(text)


# 2. Extract symptom
entities = extractor.extract(
    result["tokens"],
    result["labels"],
)


print("\nTEXT:")
print(text)

print("\nTOKENS:")
for token, label in zip(
    result["tokens"],
    result["labels"],
):
    print(f"{token:20} {label}")


print("\nEXTRACTED:")
print(entities)