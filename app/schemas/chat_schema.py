from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )


class DepartmentScore(BaseModel):
    department: str
    score: float
    diseases: list[str]


class DepartmentTriageResponse(BaseModel):
    suggested_department: str | None
    score: float
    departments: list[DepartmentScore]
    status: str


class ChatResponse(BaseModel):
    conversation_id: str
    status: str
    message: str
    symptoms: list[str]
    finished: bool
    llm_used: bool
    warning: str | None = None
    department_triage: DepartmentTriageResponse | None = None
