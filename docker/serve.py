"""Warm the local model before opening the API; keep one conversation worker."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.controller.chat_router import _get_ner
from app.main import app
import uvicorn


if __name__ == "__main__":
    _get_ner()
    print("Local NER model is ready.", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8002")))
