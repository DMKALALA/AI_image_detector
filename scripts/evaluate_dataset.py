#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.three_method_detection_service import get_detection_service  # noqa: E402


def iter_images(root_dir):
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
        for path in Path(root_dir).rglob(f"*{ext}"):
            if path.is_file():
                yield path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate detector accuracy on a labeled dataset."
    )
    parser.add_argument("--ai-dir", required=True, help="Folder with AI images.")
    parser.add_argument("--real-dir", required=True, help="Folder with real images.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap per class (0 = no limit).",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.jsonl",
        help="Write per-image results as JSONL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    service = get_detection_service()
    if service is None:
        raise SystemExit(
            "Detection service unavailable. Set ENABLE_MODEL_IMPORTS=1 and ensure "
            "PyTorch is available."
        )

    datasets = [("ai", args.ai_dir, 1), ("real", args.real_dir, 0)]
    results = []
    total = 0
    for label_name, root_dir, label in datasets:
        paths = list(iter_images(root_dir))
        if args.limit > 0:
            paths = paths[: args.limit]
        for path in paths:
            total += 1
            record = {
                "path": str(path),
                "label": label,
                "label_name": label_name,
                "prediction": None,
                "confidence": None,
                "method": None,
                "error": None,
            }
            try:
                result = service.detect_ai_image(str(path))
                if "error" in result:
                    record["error"] = result.get("error")
                else:
                    record["prediction"] = 1 if result.get("is_ai_generated") else 0
                    record["confidence"] = float(result.get("confidence", 0.0))
                    record["method"] = result.get("method")
            except Exception as exc:
                record["error"] = str(exc)

            results.append(record)

    successes = [r for r in results if r["prediction"] is not None]
    if not successes:
        raise SystemExit("No successful detections. Check logs for errors.")

    y_true = [r["label"] for r in successes]
    y_pred = [r["prediction"] for r in successes]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0]).tolist()

    def ai_probability(row):
        confidence = row["confidence"] or 0.0
        return confidence if row["prediction"] == 1 else 1.0 - confidence

    brier = mean([(ai_probability(r) - r["label"]) ** 2 for r in successes])
    correct_conf = [
        r["confidence"] for r in successes if r["confidence"] is not None and r["prediction"] == r["label"]
    ]
    incorrect_conf = [
        r["confidence"] for r in successes if r["confidence"] is not None and r["prediction"] != r["label"]
    ]

    summary = {
        "total_images": total,
        "successful": len(successes),
        "skipped": total - len(successes),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix_ai_real": cm,
        "brier_score": brier,
        "avg_confidence_correct": mean(correct_conf) if correct_conf else None,
        "avg_confidence_incorrect": mean(incorrect_conf) if incorrect_conf else None,
    }

    with open(args.output, "w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote per-image results to {args.output}")


if __name__ == "__main__":
    main()
