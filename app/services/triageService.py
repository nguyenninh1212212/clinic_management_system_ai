from pathlib import Path

import joblib
from underthesea import word_tokenize
from unidecode import unidecode
from AI.triage.safety_triage import SafetyTriage
from app.schemas.triage_schema import DepartmentPrediction, TriageResponse

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "triage_svm_v4.pkl"
)
model = joblib.load(MODEL_PATH)
safety_triage = SafetyTriage()

class TriageService:
    @staticmethod
    def preprocess_text(text: str) -> str:
        text = unidecode(text.strip().lower())
        if not text:
            raise ValueError("symptom_text must not be blank")
        return word_tokenize(text, format="text")

    @staticmethod
    def predict(symptom_text: str) -> TriageResponse:
        if not isinstance(symptom_text, str):
            raise TypeError("symptom_text must be a string")

        symptom_text = symptom_text.strip()
        if not symptom_text:
            raise ValueError("symptom_text must not be blank")

        safety_result = safety_triage.assess(symptom_text)
        if safety_result is not None:
            return TriageResponse(
                primary_department="Khoa Cấp cứu",
                confidence=0.0,
                urgency="HIGH",
                top_predictions=[],
                is_multiple_symptoms=False,
                red_flags=safety_result["red_flags"],
            )

        clean_text = TriageService.preprocess_text(symptom_text)
        probs = model.predict_proba([clean_text])[0]
        
        top_3_indices = probs.argsort()[-min(3, len(probs)):][::-1]
        top_predictions = [
            DepartmentPrediction(
                department=model.classes_[i], 
                probability=round(float(probs[i]), 4)
            )
            for i in top_3_indices
        ]

        # Lấy Khoa Top 1 làm khoa chính
        primary_dept = top_predictions[0].department
        confidence = top_predictions[0].probability
        is_multiple_symptoms = False

        # Logic đa triệu chứng
        if confidence < 0.5 and top_predictions[1].probability > 0.3:
            is_multiple_symptoms = True
            primary_dept = "Khoa Khám bệnh Đa khoa"

        return TriageResponse(
            primary_department=primary_dept,
            confidence=confidence,
            urgency="NORMAL",
            top_predictions=top_predictions,
            is_multiple_symptoms=is_multiple_symptoms,
            red_flags=[],
        )
    
    @staticmethod
    def check_health() -> bool:
        return model is not None and hasattr(model, "predict_proba")