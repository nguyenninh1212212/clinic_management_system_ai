from fastapi import APIRouter, HTTPException
from app.schemas.triage_schema import TriageRequest, TriageResponse
from app.services.triageService import TriageService

router = APIRouter()

@router.get("/health")
def health_check():
    is_loaded = TriageService.check_health()
    return {"status": "ok", "model_loaded": is_loaded}

@router.post("/predict", response_model=TriageResponse)
def predict_triage(req: TriageRequest):
    symptom_text = req.symptom_text.strip()
    if not symptom_text:
        raise HTTPException(
            status_code=422,
            detail="symptom_text must not be blank",
        )
    return TriageService.predict(symptom_text)