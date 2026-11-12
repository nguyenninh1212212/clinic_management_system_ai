"""Shared department scoring and HTTP helpers; terminal output only."""
import argparse
import collections
import csv
import json
import math
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def print_metrics(metrics, rows):
    """Display results only; never persist evaluation artifacts."""
    from _terminal import show_department
    show_department(metrics)
    for row in rows:
        if row["error"] or row["cleanup_error"]:
            print(f"Row {row['row']}: {row['error']} {row['cleanup_error']}", file=sys.stderr)



def ratio(a, b):
    return a / b if b else None


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def classification(rows):
    valid = [r for r in rows if not r["error"]]
    answered = [r for r in valid if r["predicted"]]
    correct = sum(norm(r["expected"]) == norm(r["predicted"]) for r in valid)
    labels = sorted({norm(r["expected"]) for r in rows})
    per_class = {}
    confusion = collections.defaultdict(collections.Counter)
    for r in valid:
        confusion[norm(r["expected"])][norm(r["predicted"]) or "__abstain__"] += 1
    for label in labels:
        tp = sum(norm(r["expected"]) == label and norm(r["predicted"]) == label for r in valid)
        fp = sum(norm(r["expected"]) != label and norm(r["predicted"]) == label for r in valid)
        fn = sum(norm(r["expected"]) == label and norm(r["predicted"]) != label for r in valid)
        per_class[label] = {"precision": ratio(tp, tp + fp) or 0, "recall": ratio(tp, tp + fn) or 0,
                            "f1": ratio(2 * tp, 2 * tp + fp + fn) or 0,
                            "support": sum(norm(r["expected"]) == label for r in valid)}
    supported = [v for v in per_class.values() if v["support"]]
    return {"samples": len(rows), "successful_requests": len(valid), "errors": len(rows) - len(valid),
            "correct": correct, "accuracy_all_samples_errors_count_as_incorrect": ratio(correct, len(rows)),
            "accuracy_successful_requests": ratio(correct, len(valid)),
            "coverage_successful_requests": ratio(len(answered), len(valid)),
            "accuracy_when_suggested": ratio(correct, len(answered)),
            "macro_f1_successful_requests": ratio(sum(x["f1"] for x in supported), len(supported)),
            "top3_accuracy_successful_requests": ratio(sum(norm(r["expected"]) in json.loads(r["top3"]) for r in valid), len(valid)),
            "per_class": per_class, "confusion_matrix": dict(confusion),
            "latency_ms_median": percentile([r["latency_ms"] for r in valid], .5),
            "latency_ms_p95": percentile([r["latency_ms"] for r in valid], .95)}


def request_json(url, payload, timeout, method="POST"):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def department(args):
    source = read_csv(args.data)
    if args.limit:
        source = source[:args.limit]
    if not source:
        raise ValueError("Dataset is empty")
    for i, row in enumerate(source, 1):
        if not row.get(args.text_column, "").strip() or not row.get(args.label_column, "").strip():
            raise ValueError(f"Missing text or label at CSV row {i + 1}")
    results = []
    for i, sample in enumerate(source, 1):
        row = {"row": i, "text": sample[args.text_column], "expected": sample[args.label_column],
               "predicted": "", "top3": "[]", "status": "", "latency_ms": 0.0, "error": "", "cleanup_error": ""}
        session = None
        started = time.perf_counter()
        try:
            if args.prediction_column:
                if args.prediction_column not in sample:
                    raise ValueError("Prediction column is absent")
                row["predicted"] = sample[args.prediction_column].strip()
                row["top3"] = json.dumps([norm(row["predicted"])] if row["predicted"] else [])
                row["status"] = "offline"
            else:
                response = request_json(args.url, {"message": sample[args.text_column]}, args.timeout)
                session = response.get("conversation_id")
                triage = response.get("department_triage") or {}
                row["predicted"] = triage.get("suggested_department") or ""
                row["status"] = response.get("status", "")
                row["top3"] = json.dumps([norm(x["department"]) for x in triage.get("departments", [])[:3]], ensure_ascii=False)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            if session:
                try:
                    request_json(args.url.rstrip("/") + "/" + urllib.parse.quote(session, safe=""), None, args.timeout, "DELETE")
                except Exception as exc:
                    row["cleanup_error"] = str(exc)
        results.append(row)
        print(f"Evaluated {i}/{len(source)}", file=sys.stderr)
    metrics = classification(results)
    metrics.update({"task": "department", "protocol": "offline_predictions" if args.prediction_column else "single_initial_turn",
                    "dataset": str(Path(args.data).resolve()), "endpoint": None if args.prediction_column else args.url,
                    "note": "No follow-up answers are fabricated. Single-turn results do not measure completed conversations. Offline top3 contains only the supplied prediction."})
    print_metrics(metrics, results)
    return 1 if metrics["errors"] else 0

