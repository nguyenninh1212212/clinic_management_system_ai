"""Thống kê số mẫu, số nhãn, dòng trùng của dữ liệu kiểm tra; chỉ in terminal."""
from pathlib import Path
import sys

from _metrics import norm, read_csv
from _terminal import show_table

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = read_csv(ROOT / "training/dataset/symptom/test/test_data.csv")
    duplicates = len(rows) - len({tuple(sorted(row.items())) for row in rows})
    from datasets import load_from_disk
    dataset = load_from_disk(str(ROOT / "training/dataset/vietmed_ner/test"))
    labels = {label for row in dataset for label in row["labels"]}
    show_table(["Dữ liệu", "Số mẫu", "Số nhãn", "Ghi chú"], [
        ["Gợi ý chuyên khoa", len(rows), len({norm(r['label']) for r in rows}), f"{duplicates} dòng dư trùng hoàn toàn"],
        ["Nhận diện triệu chứng — test", len(dataset), len(labels), "Nhãn BIO; chỉ chấm DISEASESYMTOM"],
    ])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
