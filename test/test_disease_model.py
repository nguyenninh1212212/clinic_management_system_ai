from AI.disease.model import DiseaseModel


MODEL_PATH = (
    r"D:\project\clinic_management_system_ai"
    r"\models\triage_svm_v4.pkl"
)


model = DiseaseModel(MODEL_PATH)


text = "Tôi bị đau đầu và chóng mặt trong 3 ngày nay"


prediction = model.predict(text)

print("Prediction:")
print(prediction)


print("\nScores:")

scores = model.predict_proba(text)

for label, score in sorted(
    scores.items(),
    key=lambda item: item[1],
    reverse=True,
):
    print(
        f"{label:30} -> {score:.4f}"
    )