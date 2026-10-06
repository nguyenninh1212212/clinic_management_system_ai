import joblib
import os
from underthesea import word_tokenize
from unidecode import unidecode
from app.schemas.triage_schema import DepartmentPrediction, TriageResponse

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/triage_svm_v1.pkl")
model = joblib.load(MODEL_PATH)

class TriageService:
    @staticmethod
    def preprocess_text(text: str) -> str:
        text = unidecode(str(text).lower())
        return word_tokenize(text, format="text")

    @staticmethod
    def predict(symptom_text: str) -> TriageResponse:
        clean_text = TriageService.preprocess_text(symptom_text)
        probs = model.predict_proba([clean_text])[0]
        
        top_3_indices = probs.argsort()[-3:][::-1]
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

        # Check Red-flags (Cấp cứu)
        urgency = "NORMAL"
        red_flags = ["kho tho", "dau nguc", "mau", "ngat", "co giat"]
        if any(flag in unidecode(symptom_text.lower()) for flag in red_flags):
            urgency = "HIGH"

        return TriageResponse(
            primary_department=primary_dept,
            confidence=confidence,
            urgency=urgency,
            top_predictions=top_predictions,
            is_multiple_symptoms=is_multiple_symptoms
        )
    
    @staticmethod
    def check_health() -> bool:
        return model is not None