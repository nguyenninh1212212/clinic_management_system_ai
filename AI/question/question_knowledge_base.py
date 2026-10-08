import pandas as pd


class QuestionKnowledgeBase:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required_columns = {
            "symptom",
            "question_type",
            "question",
            "priority",
        }

        missing_columns = (
            required_columns - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        self.questions = {}

        for row in self.df.itertuples(index=False):
            symptom = row.symptom

            self.questions[symptom] = {
                "symptom": symptom,
                "question_type": row.question_type,
                "question": row.question,
                "priority": int(row.priority),
            }

    def get_question(
        self,
        symptom: str,
    ) -> dict | None:
        return self.questions.get(symptom)

    def get_question_text(
        self,
        symptom: str,
    ) -> str | None:
        item = self.get_question(symptom)

        if not item:
            return None

        return item["question"]

    def get_all_questions(self) -> dict:
        return self.questions

    def get_question_symptoms(self) -> list[str]:
        items = list(self.questions.values())

        items.sort(
            key=lambda item: item["priority"]
        )

        return [
            item["symptom"]
            for item in items
        ]