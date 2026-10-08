import pandas as pd


class ClarificationKnowledgeBase:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required_columns = {
            "id",
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
            question_type = row.question_type

            self.questions[question_type] = {
                "id": row.id,
                "question_type": question_type,
                "question": row.question,
                "priority": int(row.priority),
            }

    def get_question(
        self,
        question_type: str,
    ) -> str | None:

        item = self.questions.get(question_type)

        if not item:
            return None

        return item["question"]