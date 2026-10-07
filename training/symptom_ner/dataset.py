from pathlib import Path

from datasets import load_from_disk


def load_ner_dataset(path: str):

    dataset = load_from_disk(path)

    print("Dataset:")
    print(dataset)

    print("\nColumns:")
    print(dataset.column_names)

    print("\nTrain sample:")
    print(dataset["train"][0])

    print("\nValidation sample:")
    print(dataset["validation"][0])

    print("\nTest sample:")
    print(dataset["test"][0])

    return dataset


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[2]

    DATA_PATH = (
        BASE_DIR
        / "training"
        / "dataset"
        / "vietmed_ner"
    )

    load_ner_dataset(str(DATA_PATH))