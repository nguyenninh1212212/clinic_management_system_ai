
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=9090
ENV MODEL_DIR=/app/models/final

WORKDIR /app

# Cài PyTorch CPU để tránh tải bộ thư viện CUDA không cần thiết.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY wsgi.py .

EXPOSE 9090

CMD ["python", "wsgi.py"]
