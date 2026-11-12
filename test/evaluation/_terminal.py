

def number(value, percent=False):
    if value is None:
        return "Không xác định"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def show_table(headers, rows):
    rows = [[str(x).replace("\n", " ") for x in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    for index, row in enumerate(rows):
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
        if index == 0:
            print("-+-".join("-" * width for width in widths))


def show_department(metrics):
    show_table(["Chỉ số chuyên khoa", "Kết quả"], [
        ["Tổng số mẫu", metrics["samples"]],
        ["Request thành công", metrics["successful_requests"]],
        ["Lỗi kỹ thuật", metrics["errors"]],
        ["Gợi ý đúng", metrics["correct"]],
        ["Tỷ lệ đúng toàn bộ mẫu", number(metrics["accuracy_all_samples_errors_count_as_incorrect"], True)],
        ["Độ bao phủ (request thành công)", number(metrics["coverage_successful_requests"], True)],
        ["Tỷ lệ đúng khi có gợi ý", number(metrics["accuracy_when_suggested"], True)],
        ["Macro-F1", number(metrics["macro_f1_successful_requests"])],
        ["Top-3 (request thành công)", number(metrics["top3_accuracy_successful_requests"], True)],
        ["Latency trung vị (ms)", number(metrics["latency_ms_median"])],
        ["Latency p95 (ms)", number(metrics["latency_ms_p95"])],
    ])
    print()
    show_table(["Chuyên khoa", "Mẫu thành công", "Precision", "Recall", "F1"], [
        [label, value["support"], *[number(value[key]) if value["support"] else number(None)
                                  for key in ("precision", "recall", "f1")]]
        for label, value in metrics["per_class"].items()
    ])
    print("Đánh giá một lượt đầu. Lỗi/không gợi ý tính là sai trên toàn bộ mẫu.")
    print("Macro-F1 tính trên nhãn có mẫu thành công; latency không gồm tải mô hình và xóa phiên.")
