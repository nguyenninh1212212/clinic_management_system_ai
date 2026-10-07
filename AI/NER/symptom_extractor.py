from typing import List, Dict

from .decoder import NERDecoder


class SymptomExtractor:

    ENTITY_MAPPING = {
        "DISEASESYMTOM": "SYMPTOM",
    }

    IGNORED_LABELS = {
        "0",
    }

    SPECIAL_TOKENS = {
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
    }

    def __init__(self):
        self.decoder = NERDecoder()

    def extract(
        self,
        tokens: List[str],
        labels: List[str],
    ) -> Dict:

        filtered_tokens = []
        filtered_labels = []

        for token, label in zip(tokens, labels):

            if token in self.SPECIAL_TOKENS:
                continue

            filtered_tokens.append(token)
            filtered_labels.append(label)

        entities = self.decoder.decode(
            filtered_tokens,
            filtered_labels,
        )

        symptoms = []
        metadata = []

        for entity in entities:

            label = entity["label"]

            if label in self.ENTITY_MAPPING:

                symptoms.append({
                    "text": entity["text"],
                    "label": self.ENTITY_MAPPING[label],
                })

            elif label not in self.IGNORED_LABELS:

                metadata.append(entity)

        return {
            "symptoms": symptoms,
            "metadata": metadata,
        }