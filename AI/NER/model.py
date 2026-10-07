from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
import torch


class NERModel:
    def __init__(self, model_path: str):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

    def predict(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )

        return predictions