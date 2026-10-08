import pandas as pd


class ConfirmationKnowledgeBase:

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required_columns = {
            "id",
            "confirmation_type",
            "text",
            "priority",
        }

        missing_columns = (
            required_columns - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        self.confirmations = (
            self.df
            .sort_values("priority")
            .to_dict("records")
        )

    def get_question(self) -> str | None:

        for item in self.confirmations:

            if item["confirmation_type"] == "question":
                return str(item["text"])

        return None

    def get_confirmations(self) -> list[dict]:

        return [
            item
            for item in self.confirmations
            if item["confirmation_type"] in {
                "yes",
                "no",
            }
        ]