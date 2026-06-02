"""Ve bieu do hoc thuat tu file JSON danh gia (tieng Viet khong dau)."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_accuracy_coverage(summary_path: Path, out_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    curve = summary.get("coverage_curve", [])

    coverage = [c.get("coverage", 0) for c in curve]
    accuracy = [c.get("accuracy") for c in curve]

    plt.figure(figsize=(6, 4))
    plt.plot(coverage, accuracy, marker="o", linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrix(details_path: Path, out_path: Path) -> None:
    with details_path.open("r", encoding="utf-8") as f:
        details = json.load(f)

    accepted = [d for d in details if d.get("triage_reason") == "accepted"]
    if not accepted:
        print("Khong co mau duoc chap nhan de ve confusion matrix.")
        return

    y_true = [d.get("true_label", "UNKNOWN") for d in accepted]
    y_pred = [d.get("pred_label", "UNKNOWN") for d in accepted]
    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Accepted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ve bieu do tu danh gia")
    parser.add_argument("--summary", default=r"D:\DoAn_DaLieu\5_Results\eval_data_test_summary.json")
    parser.add_argument("--details", default=r"D:\DoAn_DaLieu\5_Results\eval_data_test_details.json")
    parser.add_argument("--out_dir", default=r"D:\DoAn_DaLieu\5_Results")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    details_path = Path(args.details)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_coverage(summary_path, out_dir / "accuracy_vs_coverage.png")
    plot_confusion_matrix(details_path, out_dir / "confusion_matrix_accepted.png")
    print("Da luu bieu do vao:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
