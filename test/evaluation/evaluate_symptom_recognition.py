"""Evaluate symptom text sets against the bundled VietMed NER test split."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def gold_symptoms(words, labels):
    if len(words) != len(labels):
        raise ValueError("Mismatched words and BIO labels")
    entities, current = [], []
    for word, label in zip(words, labels):
        if label != "I-DISEASESYMTOM" and current:
            entities.append(" ".join(current))
            current = []
        if label in {"B-DISEASESYMTOM", "I-DISEASESYMTOM"}:
            current.append(word)
    if current:
        entities.append(" ".join(current))
    return entities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "training/dataset/vietmed_ner/test")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    from datasets import load_from_disk
    dataset = load_from_disk(str(args.data))
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    samples = []
    for i in range(count):
        row = dataset[i]
        if " ".join(row["text"].split()) != " ".join(" ".join(row["words"]).split()):
            raise ValueError(f"Text and gold words differ at row {i}")
        samples.append({"text": row["text"], "expected_symptoms": gold_symptoms(row["words"], row["labels"])})
    if not samples:
        raise ValueError("Dataset is empty")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    from AI.NER.inference import NERInference
    from AI.NER.symptom_extractor import SymptomExtractor
    from _metrics import norm, ratio
    model = NERInference(str(ROOT / "models/symptom_ner_model/final"))
    extractor = SymptomExtractor()
    tp = fp = fn = exact = 0
    print(f"Đánh giá triệu chứng: {count}/{len(dataset)} câu", flush=True)
    for i, sample in enumerate(samples, 1):
        result = model.predict(sample["text"])
        predicted = extractor.extract(result["tokens"], result["labels"])["symptoms"]
        gold = set(map(norm, sample["expected_symptoms"]))
        pred = {norm(x["text"]) for x in predicted}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += gold == pred
        if i % 100 == 0:
            print(f"Đã chấm {i}/{count}", flush=True)
    from _terminal import show_table, number
    show_table(["Chỉ số", "Kết quả"], [
        ["Số câu đánh giá", count], ["TP", tp], ["FP", fp], ["FN", fn],
        ["Precision", number(ratio(tp, tp + fp))],
        ["Recall", number(ratio(tp, tp + fn))],
        ["Micro-F1", number(ratio(2 * tp, 2 * tp + fp + fn))],
        ["Tỷ lệ câu khớp hoàn toàn", number(exact / count, True)],
    ])
    print("Nguồn:", args.data)
    print("So khớp tập chuỗi; không phải F1 span/BIO. Mỗi câu đọc tối đa 256 token.")
    print("Cần xác minh độc lập train/test trước khi kết luận khả năng tổng quát.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
