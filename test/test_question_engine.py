from typing import Any

from AI.pipeline.symptom_pipeline import NERPredictor
from AI.question.question_service import QuestionService


class NoEntityNer(NERPredictor):
    def predict(self, text: str) -> dict[str, Any]:
        _ = text
        return {
            "tokens": ["<s>", "</s>"],
            "labels": ["O", "O"],
        }


class FailingNer(NERPredictor):
    def predict(self, _text: str) -> dict[str, Any]:
        _ = _text
        raise AssertionError("Emergency handling must bypass NER")


def make_question_service(max_questions: int = 8) -> QuestionService:
    return QuestionService(
        model_path="models/symptom_ner_model/final",
        knowledge_base_path=(
            "training/dataset/disease/"
            "disease_knowledge_base.csv"
        ),
        ner=NoEntityNer(),
        max_questions=max_questions,
    )


def print_result(result):
    print("\n" + "=" * 70)

    print("PATIENT STATE")
    print("-" * 70)

    for code, state in result["patient_state"].symptoms.items():
        print(
            f"{code:<35} "
            f"value={str(state.value):<5} "
            f"duration={state.duration} "
            f"severity={state.severity}"
        )

    print("\nDISEASE CANDIDATES")
    print("-" * 70)

    for index, disease in enumerate(
        result["ranked_diseases"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{disease['disease']:<40} "
            f"score={disease['score']:.4f}"
        )

        print(
            f"   matched : "
            f"{', '.join(disease['matched_symptoms']) or '-'}"
        )

        print(
            f"   negative: "
            f"{', '.join(disease['negative_symptoms']) or '-'}"
        )

        print(
            f"   missing : "
            f"{', '.join(disease['missing_symptoms']) or '-'}"
        )

    print("\nNEXT QUESTION")
    print("-" * 70)

    next_question = result["next_question"]

    if next_question:
        print(
            f"symptom : {next_question['symptom']}"
        )
        print(
            f"question: {next_question['question']}"
        )
    else:
        print("None")

    print(
        f"\nfinished: {result['finished']}"
    )

    print("=" * 70)


def test_question_selection():
    service = QuestionService(
        model_path="models/symptom_ner_model/final",
        knowledge_base_path=(
            "training/dataset/disease/"
            "disease_knowledge_base.csv"
        ),
    )

    result = service.process_message(
        "Tôi bị đau đầu và chóng mặt"
    )

    print_result(result)

    assert result["next_question"] is not None
    assert result["next_question"]["symptom"] is not None
    assert result["next_question"]["question"] is not None


def test_leg_pain_first_gets_a_generic_detail_question():
    service = make_question_service()

    result = service.process_message("Tôi bị đau ở chân")

    assert result["symptoms"] == ["leg_pain"]
    assert result["next_question"]["symptom"] == "leg_pain"
    assert result["next_question"]["question_type"] == "symptom_details"
    assert "triệu chứng này" in result["next_question"]["question"]
    assert "khó thở" not in result["next_question"]["question"]


def test_dull_leg_pain_description_does_not_trigger_unclear_symptom():
    service = make_question_service()

    result = service.process_message("Tôi bị đau âm ỉ ở chân")

    assert result["status"] == "asking_question"
    assert result["symptoms"] == ["leg_pain"]
    assert result["next_question"]["symptom"] == "leg_pain"
    assert result["next_question"]["question_type"] == "symptom_details"


def test_symptom_details_are_saved_before_next_candidate_question():
    service = make_question_service()
    service.process_message("Tôi bị đau ở chân")

    result = service.process_message(
        "Đau ở bắp chân từ 3 ngày nay, đau nhiều"
    )

    state = result["patient_state"].symptoms["leg_pain"]
    assert state.value is True
    assert state.duration == "từ 3 ngày"
    assert state.severity == "severe"
    assert result["next_question"]["question_type"] != "symptom_details"


def test_detail_question_is_generic_for_other_symptoms_too():
    service = make_question_service()

    result = service.process_message("Tôi bị đau đầu")

    assert result["next_question"]["symptom"] == "headache"
    assert result["next_question"]["question_type"] == "symptom_details"
    assert "triệu chứng này" in result["next_question"]["question"]


def test_question_conversation_flow():
    service = QuestionService(
        model_path="models/symptom_ner_model/final",
        knowledge_base_path=(
            "training/dataset/disease/"
            "disease_knowledge_base.csv"
        ),
    )

    result = service.process_message(
        "Tôi bị đau đầu và chóng mặt"
    )

    assert result["next_question"] is not None

    print("\nTURN 1:")
    print_result(result)

    for i in range(3):
        question = result["next_question"]

        print(f"\nTURN {i + 2}:")
        print(
            f"QUESTION: {question['question']}"
        )

        result = service.process_message("Có")

        print_result(result)

        if result["finished"]:
            break

    assert result["patient_state"] is not None


def test_question_negative_answer():
    service = QuestionService(
        model_path="models/symptom_ner_model/final",
        knowledge_base_path=(
            "training/dataset/disease/"
            "disease_knowledge_base.csv"
        ),
    )

    result = service.process_message(
        "Tôi bị đau đầu và chóng mặt"
    )

    assert result["next_question"] is not None

    print("\nTURN 1:")
    print_result(result)

    for i in range(3):
        question = result["next_question"]

        print(f"\nTURN {i + 2}:")
        print(
            f"QUESTION: {question['question']}"
        )

        result = service.process_message("Không")

        print_result(result)

        if result["finished"]:
            break
    print("\nDepartment Triage:")

    print(
        result["department_triage"]
    )
    assert result["patient_state"] is not None


def test_short_negative_answer_is_not_treated_as_clarification():
    service = QuestionService(
        model_path="models/symptom_ner_model/final",
        knowledge_base_path=(
            "training/dataset/disease/"
            "disease_knowledge_base.csv"
        ),
    )

    service.current_question = {
        "symptom": "shortness_of_breath",
        "question_type": "yes_no",
        "question": "Bạn có cảm thấy khó thở hoặc hụt hơi không?",
    }

    result = service.process_message("không, tôi không bị")

    assert result["status"] != "needs_clarification"
    assert result["next_question"] is not None
    assert result["patient_state"].symptoms["shortness_of_breath"].value is False


def test_emergency_message_bypasses_routine_question_flow():
    service = QuestionService(
        model_path="models/symptom_ner_model/final",
        knowledge_base_path=(
            "training/dataset/disease/"
            "disease_knowledge_base.csv"
        ),
        ner=FailingNer(),
    )
    service.current_question = {
        "symptom": "headache",
        "question_type": "yes_no",
        "question": "Bạn có đau đầu không?",
    }

    result = service.process_message(
        "Tôi đột ngột yếu liệt một bên người"
    )

    assert result["status"] == "emergency"
    assert result["finished"] is False
    assert result["next_question"] is None
    assert result["safety"]["urgency"] == "HIGH"
    assert "possible_sudden_neurological_deficit" in (
        result["safety"]["red_flags"]
    )
    assert service.current_question is None
    assert result["symptoms"]


def test_safety_triage_handles_urgent_signs_and_negation():
    from AI.triage.safety_triage import SafetyTriage

    safety = SafetyTriage()

    cases = (
        ("Tôi khó thở dữ dội", "severe_breathing_difficulty"),
        ("Tôi không thở nổi", "severe_breathing_difficulty"),
        ("Tôi đau ngực dữ dội", "severe_or_complicated_chest_pain"),
        ("Tôi đau ngực nặng", "severe_or_complicated_chest_pain"),
        ("Tôi đột ngột yếu một bên", "possible_sudden_neurological_deficit"),
        ("Tôi bị co giật", "loss_of_consciousness_seizure_or_severe_bleeding"),
    )
    for text, expected_flag in cases:
        result = safety.assess(text)
        assert result is not None
        assert expected_flag in result["red_flags"]

    assert safety.assess("Tôi không khó thở, chỉ bị sổ mũi") is None
    assert safety.assess("Tôi không đau ngực, không ngất") is None
    assert safety.assess("Tôi đau ngực nhẹ, không khó thở") is None
    assert safety.assess("Tôi đau ngực nhẹ, khó thở") is not None


def test_current_answer_also_updates_other_positive_and_negative_symptoms():
    service = make_question_service()
    service.symptom_pipeline.patient_state.update_symptom(
        code="headache",
        value=True,
    )
    service.current_question = {
        "symptom": "dizziness",
        "question_type": "yes_no",
        "question": "Bạn có chóng mặt không?",
    }

    result = service.process_message(
        "Tôi không chóng mặt nhưng đau ngực."
    )

    state = result["patient_state"].symptoms
    assert state["headache"].value is True
    assert state["dizziness"].value is False
    assert state["chest_pain"].value is True


def test_new_free_form_symptoms_accumulate_and_reset_per_patient():
    service = make_question_service()

    service.process_message("Tôi bị đau đầu.")
    service.process_message("Tôi bị tê tay và sổ mũi.")

    state = service.symptom_pipeline.patient_state.get_state()
    assert state.symptoms["headache"].value is True
    assert state.symptoms["paresthesia"].value is True
    assert state.symptoms["coryza"].value is True

    service.reset()

    assert service.symptom_pipeline.patient_state.get_state().symptoms == {}
    assert service.current_question is None


def test_initial_free_form_description_preserves_each_symptom_polarity():
    service = make_question_service()

    result = service.process_message(
        "Tôi không chóng mặt nhưng hơi đau ngực."
    )

    state = result["patient_state"].symptoms
    assert state["dizziness"].value is False
    assert state["chest_pain"].value is True


def test_max_questions_is_enforced_and_has_explicit_limit_status():
    service = make_question_service(max_questions=1)

    first = service.process_message("Tôi bị sốt và ho.")

    assert first["status"] == "asking_question"
    assert first["questions_asked"] == 1
    assert first["next_question"]["question_type"] == "yes_no"

    capped = service.process_message("Có")

    assert capped["status"] == "awaiting_confirmation"
    assert capped["questions_asked"] == 1
    assert capped["next_question"]["question_type"] == "confirmation"
    assert service.current_question is None

    stopped = service.process_message("Chưa đủ")

    assert stopped["status"] == "question_limit_reached"
    assert stopped["next_question"] is None
    assert stopped["finished"] is False


def test_max_questions_also_stops_repeated_unrecognized_clarification():
    service = make_question_service(max_questions=1)

    first = service.process_message("Tôi thấy không ổn.")
    assert first["questions_asked"] == 1
    assert first["next_question"]["question_type"] == "unclear_symptom"

    capped = service.process_message("Tôi không biết.")
    assert capped["status"] == "awaiting_confirmation"
    assert capped["questions_asked"] == 1

    stopped = service.process_message("không rõ")
    assert stopped["status"] == "question_limit_reached"
    assert stopped["next_question"] is None


def test_reset_clears_question_limit_counter():
    service = make_question_service(max_questions=1)
    service.process_message("Tôi bị sốt và ho.")

    service.reset()

    assert service.questions_asked == 0
    first = service.process_message("Tôi bị sốt và ho.")
    assert first["questions_asked"] == 1
    assert first["next_question"]["question_type"] == "yes_no"


def test_confirmation_and_department_confirmation_have_distinct_states():
    service = make_question_service(max_questions=1)
    service.process_message("Tôi bị sốt và ho.")

    confirmation = service.process_message("Có")
    assert confirmation["status"] == "awaiting_confirmation"

    department_confirmation = service.process_message("Đủ rồi")
    assert department_confirmation["status"] == (
        "awaiting_department_confirmation"
    )
    assert (
        department_confirmation["next_question"]["question_type"]
        == "department_confirmation"
    )

    completed = service.process_message("Đồng ý")
    assert completed["status"] == "completed"
    assert completed["finished"] is True