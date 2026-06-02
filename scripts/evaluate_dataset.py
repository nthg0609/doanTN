"""Cong cu danh gia he thong tren tap Data_test (tieng Viet)."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm

# --- [BẮT BUỘC] PHẢI ÉP ĐƯỜNG DẪN TRƯỚC KHI IMPORT BẤT KỲ MODULE NỘI BỘ NÀO ---
BASE_DIR = Path("d:/DoAn_DaLieu")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# -----------------------------------------------------------------------------

# Bây giờ Python mới có thể tìm thấy thư mục pipeline
from pipeline.unified_pipeline import UnifiedDermatologyPipeline
from pipeline.safety_gate import SafetyGate, SafetyGateConfig


def _list_images(root_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for path in root_dir.rglob("*"):
        if path.suffix.lower() in exts:
            images.append(path)
    return images


def _label_from_path(path: Path) -> str:
    return path.parent.name.strip().upper()


def _build_mask_index(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image_path") or row.get("image") or ""
            mask = row.get("mask_path") or row.get("mask") or ""
            if img and mask:
                mapping[Path(img).name] = mask
    return mapping


def _find_mask_path(img_path: Path, mask_index: Dict[str, str]) -> Optional[Path]:
    # Uu tien mapping tu test.csv
    if img_path.name in mask_index:
        candidate = Path(mask_index[img_path.name])
        if candidate.exists():
            return candidate

    # Tim theo quy uoc trong thu muc
    stem = img_path.stem
    parent = img_path.parent
    candidates = [
        parent / f"{stem}_segmentation.png",
        parent / f"{stem}_mask.png",
        parent / f"{stem}.png",
        parent / "masks" / f"{stem}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_mask(mask_path: Path) -> Optional[np.ndarray]:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.uint8)


def _align_pred(pred: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if pred.shape != target_shape:
        pred = cv2.resize(pred.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (pred > 0).astype(np.uint8)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return 1.0 if union == 0 and inter == 0 else float(inter) / float(union + 1e-8)


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gt).sum())
    return (2.0 * inter) / float(pred.sum() + gt.sum() + 1e-8)


def _coverage_metrics(y_true, y_pred, thresholds, confidences, mask_ok):
    curve = []
    for tau in thresholds:
        accepted_idx = [i for i, (c, ok) in enumerate(zip(confidences, mask_ok)) if ok and c is not None and c >= tau]
        if not accepted_idx:
            curve.append({"tau": tau, "coverage": 0.0, "accuracy": None, "f1_macro": None, "precision_macro": None, "recall_macro": None})
            continue
        y_true_a = [y_true[i] for i in accepted_idx]
        y_pred_a = [y_pred[i] for i in accepted_idx]
        acc = accuracy_score(y_true_a, y_pred_a)
        f1 = f1_score(y_true_a, y_pred_a, average="macro", zero_division=0)
        prec = precision_score(y_true_a, y_pred_a, average="macro", zero_division=0)
        rec = recall_score(y_true_a, y_pred_a, average="macro", zero_division=0)
        coverage = len(accepted_idx) / max(1, len(y_true))
        curve.append({
            "tau": float(tau),
            "coverage": float(coverage),
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
        })
    return curve


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Danh gia he thong tren tap Data_test")
    parser.add_argument("--data_dir", default=r"D:\DoAn_DaLieu\1_Data\Data_test", help="Thu muc Data_test")
    parser.add_argument("--output", default=r"D:\DoAn_DaLieu\5_Results\eval_data_test_summary.json", help="File tong ket")
    parser.add_argument("--details", default=r"D:\DoAn_DaLieu\5_Results\eval_data_test_details.json", help="File chi tiet")
    parser.add_argument("--safety_config", default=r"D:\DoAn_DaLieu\config\safety_gate.json", help="Cau hinh Safety Gate")
    parser.add_argument("--min_conf", type=float, default=None, help="Ghi de nguong tu choi")
    parser.add_argument("--limit", type=int, default=None, help="Gioi han so anh de test nhanh")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {data_dir}")

    # Nap cau hinh Safety Gate
    if args.safety_config and Path(args.safety_config).exists():
        config = SafetyGateConfig.from_json(args.safety_config)
    else:
        config = SafetyGateConfig()
    if args.min_conf is not None:
        config = SafetyGateConfig.from_dict({**config.__dict__, "min_class_confidence": args.min_conf})

    # Gate mac dinh va gate chi danh gia mask
    gate_default = SafetyGate(config)
    mask_only_cfg = SafetyGateConfig.from_dict({**config.__dict__, "min_class_confidence": 0.0})
    gate_mask_only = SafetyGate(mask_only_cfg)

    pipeline = UnifiedDermatologyPipeline(
        safety_config=config,
        mode="classification",
        load_models=True,
    )

    images = _list_images(data_dir)
    if args.limit:
        images = images[: args.limit]

    # Lap chi muc mask tu test.csv (neu co)
    test_csv = Path(r"D:\DoAn_DaLieu\1_Data\processed\segmentation\test.csv")
    mask_index = _build_mask_index(test_csv)

    y_true = []
    y_pred = []
    confidences = []
    mask_ok = []
    triage_reasons = []
    iou_list = []
    dice_list = []
    details = []

    for img_path in tqdm(images, desc="Dang danh gia"):
        true_label = _label_from_path(img_path)
        result = pipeline.run(str(img_path), return_mask=True)

        cls = result.get("classification") or {}
        pred_label = cls.get("prediction")
        conf = cls.get("confidence")
        metrics = result.get("metrics") or {}

        gate_result = gate_default.evaluate(metrics, conf)
        mask_result = gate_mask_only.evaluate(metrics, conf)

        y_true.append(true_label)
        y_pred.append(pred_label if pred_label is not None else "UNKNOWN")
        confidences.append(conf)
        mask_ok.append(mask_result.accept)
        triage_reasons.append(gate_result.reason if not gate_result.accept else "accepted")

        # Tinh IoU/Dice neu co mask GT
        gt_path = _find_mask_path(img_path, mask_index)
        pred_mask = result.get("segmentation_mask")
        if gt_path and pred_mask is not None:
            gt_mask = _load_mask(gt_path)
            if gt_mask is not None:
                pred_bin = _align_pred(pred_mask, gt_mask.shape)
                iou_list.append(_iou(pred_bin, gt_mask))
                dice_list.append(_dice(pred_bin, gt_mask))

        details.append({
            "image": str(img_path),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": conf,
            "triage_reason": triage_reasons[-1],
            "mask_ok": mask_result.accept,
        })

    total = len(images)
    triage_count = sum(1 for r in triage_reasons if r != "accepted")
    accept_count = total - triage_count

    # Coverage curve theo tau
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    curve = _coverage_metrics(y_true, y_pred, thresholds, confidences, mask_ok)

    # Accuracy/F1 tren tap duoc chap nhan (theo cau hinh gate)
    accepted_idx = [i for i, reason in enumerate(triage_reasons) if reason == "accepted"]
    if accepted_idx:
        y_true_a = [y_true[i] for i in accepted_idx]
        y_pred_a = [y_pred[i] for i in accepted_idx]
        acc_a = accuracy_score(y_true_a, y_pred_a)
        f1_a = f1_score(y_true_a, y_pred_a, average="macro", zero_division=0)
    else:
        acc_a = None
        f1_a = None

    summary = {
        "total_images": total,
        "triage_rate": triage_count / max(1, total),
        "accept_rate": accept_count / max(1, total),
        "acc_on_accepted": acc_a,
        "f1_macro_on_accepted": f1_a,
        "mean_iou": float(np.mean(iou_list)) if iou_list else None,
        "mean_dice": float(np.mean(dice_list)) if dice_list else None,
        "num_masks_for_iou": len(iou_list),
        "coverage_curve": curve,
        "safety_config": config.__dict__,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(args.details, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print("\n[OK] Hoan tat danh gia. Luu summary tai:", args.output)
    print("[OK] Luu chi tiet tai:", args.details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())