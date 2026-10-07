import os
import joblib


class DiseaseModel:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Disease model not found: {model_path}"
            )

        self.model = joblib.load(model_path)

    def predict(self, text: str) -> str:
        return self.model.predict([text])[0]

    def predict_proba(self, text: str):
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                "Disease model does not support predict_proba()."
            )

        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.classes_

        return {
            label: float(probability)
            for label, probability in zip(
                classes,
                probabilities,
            )
        }