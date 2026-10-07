# training/symptom_ner/evaluate.py

from pathlib import Path

import numpy as np

from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
)

from dataset import (
    create_dataset,
    LABELS,
)


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "vietmed_ner"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "symptom_ner"
)


def main():

    dataset, tokenizer = create_dataset(
        str(DATA_PATH)
    )

    split = dataset.train_test_split(
        test_size=0.2,
        seed=42
    )

    test_dataset = split["test"]

    model = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_PATH)
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
    )

    predictions = trainer.predict(
        test_dataset
    )

    logits = predictions.predictions
    labels = predictions.label_ids

    pred_ids = np.argmax(
        logits,
        axis=-1
    )

    true_labels = []
    pred_labels = []

    for prediction, label in zip(
        pred_ids,
        labels
    ):

        current_true = []
        current_pred = []

        for pred_id, label_id in zip(
            prediction,
            label
        ):

            # -100 = special token
            if label_id == -100:
                continue

            current_true.append(
                LABELS[label_id]
            )

            current_pred.append(
                LABELS[pred_id]
            )

        true_labels.append(
            current_true
        )

        pred_labels.append(
            current_pred
        )

    precision = precision_score(
        true_labels,
        pred_labels
    )

    recall = recall_score(
        true_labels,
        pred_labels
    )

    f1 = f1_score(
        true_labels,
        pred_labels
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            true_labels,
            pred_labels
        )
    )


if __name__ == "__main__":
    main()