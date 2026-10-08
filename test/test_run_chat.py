import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))

from AI.question.question_service import QuestionService


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "symptom_ner_model"
    / "final"
)

DISEASE_KB_PATH = (
    BASE_DIR
    / "training"
    / "dataset"
    / "disease"
    / "disease_knowledge_base.csv"
)


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)

    print("PATIENT STATE")
    print("=" * 60)

    state = result["patient_state"]

    for code, symptom in state.symptoms.items():
        print(
            f"- {code}: "
            f"value={symptom.value}, "
            f"duration={symptom.duration}, "
            f"severity={symptom.severity}"
        )

    print("\n" + "=" * 60)
    print("DISEASE CANDIDATES")
    print("=" * 60)

    ranked_diseases = result.get(
        "ranked_diseases",
        [],
    )

    if not ranked_diseases:
        print("Không có disease candidate.")
    else:
        for index, item in enumerate(
            ranked_diseases,
            start=1,
        ):
            print(
                f"{index}. "
                f"{item['disease']} "
                f"score={item['score']}"
            )

    print("\n" + "=" * 60)
    print("DEPARTMENT TRIAGE")
    print("=" * 60)

    department_triage = result.get(
        "department_triage"
    )

    if department_triage:
        print(
            "Suggested department:",
            department_triage.get(
                "suggested_department"
            ),
        )

        print(
            "Score:",
            department_triage.get("score"),
        )

    print("\n" + "=" * 60)
    print("NEXT QUESTION")
    print("=" * 60)

    next_question = result.get(
        "next_question"
    )

    if next_question:
        print(
            next_question["question"]
        )
    else:
        print("Không còn câu hỏi.")

    print("\nFinished:", result["finished"])


def main():
    service = QuestionService(
        model_path=str(MODEL_PATH),
        knowledge_base_path=str(
            DISEASE_KB_PATH
        ),
    )

    print("=" * 60)
    print("AI CLINIC CHAT TERMINAL")
    print("=" * 60)

    print(
        "Nhập triệu chứng hoặc trả lời câu hỏi của AI."
    )
    print("Gõ 'exit' để thoát.")
    print("Gõ 'reset' để bắt đầu bệnh nhân mới.")

    while True:
        print()

        text = input("Patient: ").strip()

        if not text:
            continue

        if text.lower() == "exit":
            break

        if text.lower() == "reset":
            service.reset()
            print("Đã reset patient state.")
            continue

        try:
            result = service.process_message(text)

            print_result(result)

        except Exception as error:
            print(
                "\nERROR:",
                type(error).__name__,
                error,
            )


if __name__ == "__main__":
    main()