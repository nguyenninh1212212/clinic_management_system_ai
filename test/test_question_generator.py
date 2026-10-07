from AI.question.question_generator import QuestionGenerator


generator = QuestionGenerator()


print("=" * 60)
print("QUESTION GENERATOR TEST")
print("=" * 60)

test_symptoms = [
    "headache",
    "dizziness",
    "vomiting",
    "chest_pain",
    "shortness_of_breath",
]

for symptom in test_symptoms:
    question = generator.generate(symptom)

    print(f"\nSymptom: {symptom}")
    print(f"Question: {question}")

print("\n" + "=" * 60)
print("TEST FINISHED")
print("=" * 60)
