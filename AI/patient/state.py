from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SymptomState:
    value: Optional[bool] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    location: Optional[str] = None


@dataclass
class PatientState:
    symptoms: dict[str, SymptomState] = field(
        default_factory=dict
    )