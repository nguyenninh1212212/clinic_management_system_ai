from pydantic import BaseModel, Field

class DepartmentPrediction(BaseModel):
    department: str
    probability: float = Field(
        description=(
            "Raw classifier score; not clinically calibrated and not a "
            "disease probability."
        )
    )

class TriageRequest(BaseModel):
    symptom_text: str = Field(min_length=1, max_length=2000)

class TriageResponse(BaseModel):
    primary_department: str
    confidence: float = Field(
        description=(
            "Classifier output score for the top model prediction; not "
            "clinically calibrated. When is_multiple_symptoms is true, "
            "primary_department is a general-routing fallback. Emergency "
            "responses use 0 because the classifier is bypassed."
        )
    )
    urgency: str
    top_predictions: list[DepartmentPrediction]
    is_multiple_symptoms: bool
    red_flags: list[str] = Field(default_factory=list)