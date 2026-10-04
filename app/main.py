from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from underthesea import word_tokenize

app = FastAPI(title="AI Triage API", version="1.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/triage_svm_v1.pkl")
model = joblib.load(MODEL_PATH)

class TriageRequest(BaseModel):
    symptom_text: str

class TriageResponse(BaseModel):
    department: str
    confidence: float
    urgency: str

def preprocess_text(text: str):
    return word_tokenize(text.lower(), format="text")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/api/v1/triage/predict", response_model=TriageResponse)
def predict_triage(req: TriageRequest):
    clean_text = preprocess_text(req.symptom_text)

    probs = model.predict_proba([clean_text])[0]
    max_prob_index = probs.argmax()
    confidence = probs[max_prob_index]
    department = model.classes_[max_prob_index]

    urgency = "NORMAL"
    red_flags = ["khó thở", "đau ngực", "máu", "ngất", "co giật"]
    if any(flag in req.symptom_text.lower() for flag in red_flags):
        urgency = "HIGH"

    return TriageResponse(
        department=department,
        confidence=round(float(confidence), 4),
        urgency=urgency
    )
