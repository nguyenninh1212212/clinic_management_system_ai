"""Đánh giá gợi ý chuyên khoa qua API riêng; chỉ in kết quả terminal."""
import argparse
from collections import deque
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from _metrics import department, request_json

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def local_api(timeout):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    logs = deque(maxlen=30)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    def drain():
        for line in process.stdout:
            logs.append(line.rstrip())
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    base = f"http://127.0.0.1:{port}"
    try:
        print("Đang khởi động API riêng và tải mô hình...", flush=True)
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError("API không khởi động được")
            try:
                with urllib.request.urlopen(base + "/openapi.json", timeout=2) as response:
                    json.load(response)
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Hết thời gian chờ API")
                time.sleep(1)
        url = base + "/api/v1/chat"
        response = request_json(url, {"message": "Tôi bị đau đầu"}, timeout)
        if not response.get("status") or not response.get("conversation_id"):
            raise RuntimeError("API trả dữ liệu không hợp lệ")
        request_json(url + "/" + response["conversation_id"], None, timeout, "DELETE")
        yield url
    except Exception:
        print("\nLog API gần nhất:\n" + "\n".join(logs), file=sys.stderr)
        raise
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=2)
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data", default=str(ROOT / "training/dataset/symptom/test/test_data.csv"))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.text_column, args.label_column = "text_vi", "label"
    args.prediction_column = None
    try:
        with local_api(args.timeout) as url:
            args.url = url
            return department(args)
    except Exception as exc:
        print(f"Đánh giá thất bại: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
