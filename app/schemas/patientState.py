from pydantic import BaseModel, Field
from typing import Optional, List

class Symptom(BaseModel):
    name: str
    severity: Optional[int] = Field(default=None, ge=0, le=10)
    location: Optional[str] = None
    duration: Optional[str] = None
    onset: Optional[str] = None


class PatientState(BaseModel):
    chief_complaint: Optional[str] = None
    symptoms: List[Symptom] = []

    diet: List[str] = []
    habits: List[str] = []
    medications: List[str] = []

    medical_history: List[str] = []
    allergies: List[str] = []

    red_flags: List[str] = []

    possible_conditions: List[str] = []
    recommended_department: Optional[str] = None

    missing_information: List[str] = []
    is_complete: bool = False