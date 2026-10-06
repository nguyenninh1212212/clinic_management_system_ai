from fastapi import APIRouter
from app.schemas.triage_schema import TriageRequest, TriageResponse
from app.services.triageService import TriageService

router = APIRouter()

@router.get("/health")
def health_check():
    is_loaded = TriageService.check_health()
    return {"status": "ok", "model_loaded": is_loaded}

@router.post("/predict", response_model=TriageResponse)
def predict_triage(req: TriageRequest):
    return TriageService.predict(req.symptom_text)