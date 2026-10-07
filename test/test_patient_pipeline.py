from AI.pipeline.symptom_pipeline import SymptomPipeline


MODEL_PATH = (
    r"D:\project\clinic_management_system_ai"
    r"\models\symptom_ner_model\final"
)


pipeline = SymptomPipeline(
    model_path=MODEL_PATH
)


text = "Tôi bị đau chân và chóng mặt trong 9 ngày nay, sau khi tôi ăn 1 loại quả lạ nào đó"


result = pipeline.process(text)


print("\nTEXT:")
print(result["text"])


print("\nENTITIES:")
print(result["entities"])


print("\nNORMALIZED SYMPTOMS:")
print(result["symptoms"])


print("\nPATIENT STATE:")

state = result["patient_state"]

for code, symptom in state.symptoms.items():

    print(
        code,
        "->",
        {
            "value": symptom.value,
            "duration": symptom.duration,
            "severity": symptom.severity,
            "location": symptom.location,
        },
    )