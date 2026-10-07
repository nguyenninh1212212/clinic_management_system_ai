from typing import List, Dict


class NERDecoder:

    @staticmethod
    def decode(
        tokens: List[str],
        labels: List[str],
    ) -> List[Dict[str, str]]:

        entities = []

        current_tokens = []
        current_label = None

        for token, label in zip(tokens, labels):

            # B-XXX
            if label.startswith("B-"):

                # Lưu entity trước đó
                if current_tokens:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "label": current_label,
                    })

                current_tokens = [token]
                current_label = label[2:]

            # I-XXX
            elif label.startswith("I-"):

                entity_label = label[2:]

                if (
                    current_tokens
                    and current_label == entity_label
                ):
                    current_tokens.append(token)

                else:
                    if current_tokens:
                        entities.append({
                            "text": " ".join(current_tokens),
                            "label": current_label,
                        })

                    current_tokens = [token]
                    current_label = entity_label

            # O / 0
            else:

                if current_tokens:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "label": current_label,
                    })

                    current_tokens = []
                    current_label = None

        # Entity cuối
        if current_tokens:
            entities.append({
                "text": " ".join(current_tokens),
                "label": current_label,
            })

        return entities