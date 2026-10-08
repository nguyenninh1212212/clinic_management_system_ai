import pandas as pd


class DiseaseDepartmentKnowledgeBase:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required_columns = {
            "disease",
            "department",
            "status",
        }

        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        self.mapping = {}

        for row in self.df.itertuples(index=False):
            disease = row.disease
            department = row.department
            status = row.status

            self.mapping[disease] = {
                "department": department,
                "status": status,
            }

    def get_department(
        self,
        disease: str,
    ) -> str | None:
        item = self.mapping.get(disease)

        if not item:
            return None

        if item["status"] != "auto":
            return None

        return item["department"]

    def get_mapping(
        self,
        disease: str,
    ) -> dict | None:
        return self.mapping.get(disease)

    def get_all_mappings(self) -> dict:
        return self.mapping