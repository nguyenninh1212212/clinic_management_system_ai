# Đánh giá AI

Các tool mới nằm trong [evaluation_tool](evaluation_tool/README.md).
Tên file nêu rõ mục đích. Kết quả chỉ hiển thị trên terminal, không tạo báo cáo.

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluation_tool/evaluate_department_suggestion.py
.\.venv\Scripts\python.exe -X utf8 evaluation_tool/evaluate_symptom_recognition.py
.\.venv\Scripts\python.exe -X utf8 evaluation_tool/audit_evaluation_datasets.py
```

Thêm `--limit 5` vào lệnh đánh giá để chạy thử. Không cần tự khởi động API.
