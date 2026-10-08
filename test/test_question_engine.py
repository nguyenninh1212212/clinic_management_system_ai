from AI.question.question_service import QuestionService


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