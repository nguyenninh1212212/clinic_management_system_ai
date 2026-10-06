from pydantic import BaseModel
from typing import List

class DepartmentPrediction(BaseModel):
    department: str
    probability: float

class TriageRequest(BaseModel):
    symptom_text: str

class TriageResponse(BaseModel):
    primary_department: str
    confidence: float
    urgency: str
    top_predictions: List[DepartmentPrediction]
    is_multiple_symptoms: bool