
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from label_studio_ml.model import LabelStudioMLBase


class PhoBERTNERModel(LabelStudioMLBase):
    """Label Studio ML backend for token-level BIO NER."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_dir = os.getenv(
            "MODEL_DIR",
            "/models/sympti/final",
        )

        if not Path(model_dir, "config.json").is_file():
            raise FileNotFoundError(
                f"Không tìm thấy config.json trong MODEL_DIR={model_dir}. "
                "Hãy kiểm tra mount thư mục mô hình."
            )

        # Offset mapping cần fast tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
        )

        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "Tokenizer hiện tại không hỗ trợ offset mapping. "
                "Cần dùng fast tokenizer tương thích với checkpoint."
            )

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir
        )

        self.model.eval()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        raw_mapping = self.model.config.id2label or {}
        self.id2label = {
            int(idx): str(label)
            for idx, label in raw_mapping.items()
        }

        if not self.id2label:
            raise ValueError(
                "Checkpoint thiếu config.id2label. "
                "Cần khôi phục ánh xạ nhãn đúng từ lúc huấn luyện."
            )

        self.max_length = min(
            int(getattr(self.model.config, "max_position_embeddings", 512)),
            512,
        )

    def _label_for_id(self, idx: int) -> str:
        return self.id2label.get(idx, "O")

    @staticmethod
    def _is_outside(label: str) -> bool:
        return label in {"O", "0", "", "LABEL_0"}

    @staticmethod
    def _split_bio(label: str):
        if label.startswith("B-") or label.startswith("I-"):
            prefix, entity = label.split("-", 1)
            return prefix, entity

        # Một số checkpoint có thể dùng nhãn entity không theo BIO.
        if label and not label.startswith("LABEL_"):
            return "B", label

        return "O", None

    def _predict_spans(self, text: str) -> list[dict[str, Any]]:
        if not text or not text.strip():
            return []

        # Giữ vị trí ký tự của từng token theo khoảng trắng.
        words = [
            (match.group(), match.start(), match.end())
            for match in re.finditer(r"\S+", text)
        ]

        if not words:
            return []

        word_texts = [word[0] for word in words]

        encoded = self.tokenizer(
            word_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )

        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

        with torch.inference_mode():
            logits = self.model(**model_inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_ids = logits.argmax(dim=-1).tolist()
            token_scores = probabilities.max(dim=-1).values.tolist()

        # Chỉ lấy dự đoán của subtoken đầu tiên mỗi word.
        # Đây là giả định phù hợp khi huấn luyện BIO theo word-level labels.
        word_predictions = {}
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in word_predictions:
                continue

            word_predictions[word_id] = (
                self._label_for_id(predicted_ids[token_index]),
                float(token_scores[token_index]),
            )

        # Ghép BIO thành các thực thể và dùng offset ký tự từ văn bản gốc.
        spans = []
        active = None

        def close_active():
            nonlocal active
            if active is not None:
                spans.append(active)
                active = None

        for word_index, (word, start, end) in enumerate(words):
            if word_index not in word_predictions:
                close_active()
                continue

            label, confidence = word_predictions[word_index]

            if self._is_outside(label):
                close_active()
                continue

            prefix, entity = self._split_bio(label)
            if not entity:
                close_active()
                continue

            if prefix == "B" or (
                active is not None and active["entity"] != entity
            ):
                close_active()
                active = {
                    "entity": entity,
                    "start": start,
                    "end": end,
                    "scores": [confidence],
                }
            elif prefix == "I" and active is None:
                # BIO không hợp lệ: I- ở đầu được xử lý như B-.
                active = {
                    "entity": entity,
                    "start": start,
                    "end": end,
                    "scores": [confidence],
                }
            else:
                active["end"] = end
                active["scores"].append(confidence)

        close_active()

        results = []
        for span in spans:
            entity = span["entity"]
            if entity not in self._configured_labels():
                continue

            start = span["start"]
            end = span["end"]
            results.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "labels": [entity],
                },
                "score": sum(span["scores"]) / len(span["scores"]),
            })

        return results

    def _configured_labels(self) -> set[str]:
        # Lấy nhãn từ cấu hình Label Studio nếu SDK có cung cấp.
        try:
            config = self.label_config
            labels = re.findall(
                r'<Label\b[^>]*\bvalue=["\']([^"\']+)',
                config,
            )
            if labels:
                return set(labels)
        except Exception:
            pass

        # Fallback: lấy nhãn entity từ id2label.
        entities = set()
        for label in self.id2label.values():
            _, entity = self._split_bio(label)
            if entity:
                entities.add(entity)
        return entities

    def predict(self, tasks, context=None, **kwargs):
        predictions = []

        for task in tasks:
            text = task.get("data", {}).get("text", "")
            results = self._predict_spans(text)

            scores = [
                result["score"]
                for result in results
                if "score" in result
            ]

            predictions.append({
                "result": results,
                "score": (
                    sum(scores) / len(scores)
                    if scores else 0.0
                ),
                "model_version": "phobert-ner",
            })

        return predictions
