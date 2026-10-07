# training/symptom_ner/inference.py

from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "symptom_ner"
)


class SymptomNER:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH)
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            str(MODEL_PATH)
        )

        self.model.eval()

    def predict(self, text: str):

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=256,
        )

        offsets = encoding.pop(
            "offset_mapping"
        )

        with torch.no_grad():

            outputs = self.model(
                **encoding
            )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )[0]

        entities = []

        current_start = None
        current_end = None

        for index, prediction in enumerate(
            predictions
        ):

            label = self.model.config.id2label[
                prediction.item()
            ]

            start, end = offsets[0][index].tolist()

            if start == end:
                continue

            if label == "B-SYMPTOM":

                if current_start is not None:

                    entities.append({
                        "text": text[
                            current_start:current_end
                        ],
                        "label": "SYMPTOM",
                        "start": current_start,
                        "end": current_end,
                    })

                current_start = start
                current_end = end

            elif label == "I-SYMPTOM":

                if current_start is not None:
                    current_end = end

            else:

                if current_start is not None:

                    entities.append({
                        "text": text[
                            current_start:current_end
                        ],
                        "label": "SYMPTOM",
                        "start": current_start,
                        "end": current_end,
                    })

                    current_start = None
                    current_end = None

        # Entity cuối cùng
        if current_start is not None:

            entities.append({
                "text": text[
                    current_start:current_end
                ],
                "label": "SYMPTOM",
                "start": current_start,
                "end": current_end,
            })

        return entities


if __name__ == "__main__":

    ner = SymptomNER()

    text = (
        "Tôi bị đau đầu dữ dội, "
        "thỉnh thoảng chóng mặt "
        "và buồn nôn."
    )

    entities = ner.predict(text)

    for entity in entities:
        print(entity)