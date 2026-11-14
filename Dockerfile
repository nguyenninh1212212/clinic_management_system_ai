FROM python:3.11-slim-bookworm AS builder
WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# CPU wheels avoid downloading CUDA libraries in the default deployment.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 5 --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

FROM python:3.11-slim-bookworm AS production
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 clinic
COPY --from=builder /opt/venv /opt/venv
COPY --chown=clinic:clinic AI ./AI
COPY --chown=clinic:clinic app ./app
COPY --chown=clinic:clinic training/dataset/disease/disease_knowledge_base.csv ./training/dataset/disease/disease_knowledge_base.csv
COPY --chown=clinic:clinic training/dataset/question ./training/dataset/question
COPY --chown=clinic:clinic training/dataset/department ./training/dataset/department
COPY --chown=clinic:clinic models/symptom_ner_model/final ./models/symptom_ner_model/final
COPY --chown=clinic:clinic docker ./docker
USER clinic
EXPOSE 8002
# A single worker is required because conversation state is process-local.
CMD ["python", "docker/serve.py"]
