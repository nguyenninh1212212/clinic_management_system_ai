import pytest

from app.services import triageService
from app.services.triageService import TriageService


def test_predict_returns_ranked_departments():
    result = TriageService.predict(
        "Tôi bị đau đầu và chóng mặt trong 3 ngày nay"
    )

    assert result.primary_department
    assert 0.0 <= result.confidence <= 1.0
    assert 1 <= len(result.top_predictions) <= 3
    assert result.urgency == "NORMAL"
    assert result.red_flags == []
    assert sum(
        prediction.probability
        for prediction in result.top_predictions
    ) <= 1.0


def test_emergency_signs_bypass_the_department_model(monkeypatch):
    def unexpected_model_call(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("Emergency symptom text must bypass the model")

    monkeypatch.setattr(
        triageService.model,
        "predict_proba",
        unexpected_model_call,
    )

    result = TriageService.predict("Tôi khó thở dữ dội")

    assert result.primary_department == "Khoa Cấp cứu"
    assert result.urgency == "HIGH"
    assert result.confidence == 0.0
    assert result.top_predictions == []
    assert "severe_breathing_difficulty" in result.red_flags


@pytest.mark.parametrize(
    "text",
    [
        "Tôi không khó thở, chỉ bị sổ mũi",
        "Tôi đau ngực nhẹ",
    ],
)
def test_non_emergency_mentions_do_not_raise_urgency(text):
    result = TriageService.predict(text)

    assert result.urgency == "NORMAL"
    assert result.red_flags == []


def test_severe_chest_pain_is_urgent():
    result = TriageService.predict("Tôi đau ngực dữ dội")

    assert result.primary_department == "Khoa Cấp cứu"
    assert result.urgency == "HIGH"
    assert "severe_or_complicated_chest_pain" in result.red_flags


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_predict_rejects_blank_symptom_text(text):
    with pytest.raises(ValueError, match="must not be blank"):
        TriageService.predict(text)


def test_health_check_reports_loaded_model():
    assert TriageService.check_health() is True