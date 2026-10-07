import torch

from .model import NERModel


class NERInference:
    def __init__(self, model_path: str):
        self.ner_model = NERModel(model_path)

        self.tokenizer = self.ner_model.tokenizer
        self.model = self.ner_model.model
        self.device = self.ner_model.device
        self.id2label = self.ner_model.id2label

    def predict(self, text: str):
        encoding = self.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        prediction_ids = torch.argmax(
            outputs.logits,
            dim=-1,
        )[0]

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0]
        )

        labels = [
                self.id2label[int(prediction_id)]
            for prediction_id in prediction_ids
        ]

        return {
            "tokens": tokens,
            "labels": labels,
        }