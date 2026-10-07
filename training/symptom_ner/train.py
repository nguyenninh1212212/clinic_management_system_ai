from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)


# =========================
# CONFIG
# =========================

MODEL_NAME = "vinai/phobert-base"

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "training"
    / "dataset"
    / "vietmed_ner"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "symptom_ner"
)


# =========================
# MAIN
# =========================

def main():

    # -------------------------
    # 1. Load dataset
    # -------------------------

    dataset = load_from_disk(
        str(DATA_PATH)
    )

    print(dataset)

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Validation: {len(validation_dataset)}"
    )

    print(
        "Features:"
    )

    print(
        train_dataset.features
    )


    # -------------------------
    # 2. Get label information
    # -------------------------

    label_feature = (
        train_dataset.features["labels"]
    )

    print(
        "\nLabel feature:"
    )

    print(
        label_feature
    )

    # =========================
# GET LABEL NAMES
# =========================

    all_labels = set()

    for split in ["train", "validation"]:
        for labels in dataset[split]["labels"]:
            all_labels.update(labels)

    label_names = sorted(all_labels)

    print("\nLabel names:")
    print(label_names)

    print(
        "\nLabels:"
    )

    print(
        label_names
    )


    label2id = {
        label: index
        for index, label in enumerate(label_names)
    }

    id2label = {
        index: label
        for index, label in enumerate(label_names)
    }

    # -------------------------
    # 3. Tokenizer
    # -------------------------

    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True)

    print("Tokenizer:", type(tokenizer))
    print("Fast tokenizer:", tokenizer.is_fast)
    # -------------------------
    # 4. Tokenize dataset
    # -------------------------

    def tokenize_function(example):

        input_ids = []
        attention_mask = []
        aligned_labels = []

        for word, label in zip(
        example["words"],
        example["labels"]
    ):

        # Tokenize từng word
            tokens = tokenizer(
                word,
                add_special_tokens=False,
                truncation=False,
        )

            word_input_ids = tokens["input_ids"]

            if not word_input_ids:
                continue

        # Token đầu tiên giữ label gốc
            input_ids.extend(word_input_ids)

            attention_mask.extend(
            [1] * len(word_input_ids)
            )

        aligned_labels.append(
            label2id[label]
        )

        # Các sub-token còn lại
        # bỏ qua trong loss
        for _ in range(len(word_input_ids) - 1):
            aligned_labels.append(-100)

    # PhoBERT special tokens
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id

        input_ids = (
        [cls_token_id]
        + input_ids
        + [sep_token_id]
    )

        attention_mask = (
        [1]
        + attention_mask
        + [1]
    )

        aligned_labels = (
        [-100]
        + aligned_labels
        + [-100]
    )

    # Giới hạn sequence length
        max_length = 256

        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        aligned_labels = aligned_labels[:max_length]

        return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": aligned_labels,
    }


    train_dataset = train_dataset.map(
        tokenize_function
    )

    validation_dataset = validation_dataset.map(
        tokenize_function
    )


    # -------------------------
    # 5. Remove unnecessary columns
    # -------------------------

    columns_to_remove = [
        "words",
        "tags",
        "text",
    ]

    train_dataset = train_dataset.remove_columns(
        columns_to_remove
    )

    validation_dataset = validation_dataset.remove_columns(
        columns_to_remove
    )


    # -------------------------
    # 6. Load PhoBERT
    # -------------------------

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,

        num_labels=len(label_names),

        id2label=id2label,

        label2id=label2id,
    )


    # -------------------------
    # 7. Data collator
    # -------------------------

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )


    # -------------------------
    # 8. Training arguments
    # -------------------------

    training_args = TrainingArguments(

        output_dir=str(
            OUTPUT_DIR
        ),

        eval_strategy="epoch",

        save_strategy="epoch",

        learning_rate=2e-5,

        per_device_train_batch_size=8,

        per_device_eval_batch_size=8,

        num_train_epochs=5,

        weight_decay=0.01,

        logging_steps=20,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        save_total_limit=2,

        report_to="none",
    )


    # -------------------------
    # 9. Trainer
    # -------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,
    )


    # -------------------------
    # 10. Train
    # -------------------------

    trainer.train()


    # -------------------------
    # 11. Save model
    # -------------------------

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )


    print(
        f"\nModel saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()