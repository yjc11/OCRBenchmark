"""
Lightweight OCR detection benchmark script.

Assumptions (adjust `load_ground_truth` / `run_detector` to your data & model):
- Ground-truth file is a JSON mapping `image_name` -> list of quadrilateral
  points, e.g. [[x1, y1, x2, y2, x3, y3, x4, y4], ...].
- Images live under `--image-dir`.
- Detection model loading/inference is left as TODO placeholders.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scoring.metrics import match_boxes  # noqa: E402


Box = List[float]


def quad_to_bbox(points: Sequence[Sequence[float]]) -> Box:
    """Convert 4-point quadrilateral to axis-aligned bbox [x1, y1, x2, y2]."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_poly(poly) -> Box | None:
    """
    Normalize various polygon/bbox notations into [x1, y1, x2, y2].
    Extend this if your label format differs.
    """
    points: Iterable[Sequence[float]] | None = None

    if isinstance(poly, dict):
        if "points" in poly:
            points = poly["points"]
        elif "poly" in poly:
            points = poly["poly"]
        elif "quad" in poly:
            points = poly["quad"]
        elif "bbox" in poly:
            # Already [x1, y1, x2, y2]
            bbox = poly["bbox"]
            if len(bbox) == 4:
                return list(map(float, bbox))
    elif isinstance(poly, list):
        if len(poly) == 8:
            # Flattened [x1, y1, x2, y2, ...]
            points = list(zip(poly[0::2], poly[1::2]))
        elif len(poly) == 4 and all(isinstance(p, (int, float)) for p in poly):
            # Already bbox
            return list(map(float, poly))
        elif len(poly) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in poly):
            points = poly

    if points:
        return quad_to_bbox(points)
    return None


def load_ground_truth(label_file: Path) -> Dict[str, List[Box]]:
    """
    Load ground-truth polygons/quads into bbox dict keyed by image file name.
    Expected default: {"img.png": [[x1, y1, x2, y2, x3, y3, x4, y4], ...]}
    """
    with open(label_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt: Dict[str, List[Box]] = defaultdict(list)

    if isinstance(data, dict):
        # Either {img: [...]} or {"annotations": [...]}
        if "annotations" in data and isinstance(data["annotations"], list):
            for ann in data["annotations"]:
                img_key = ann.get("image") or ann.get("file_name") or ann.get("img_name")
                if not img_key:
                    continue
                for poly in ann.get("polys", ann.get("points", [])):
                    bbox = normalize_poly(poly)
                    if bbox:
                        gt[img_key].append(bbox)
        else:
            for img_name, polys in data.items():
                for poly in polys:
                    bbox = normalize_poly(poly)
                    if bbox:
                        gt[img_name].append(bbox)
    elif isinstance(data, list):
        # List of entries: {"image": ..., "polys": [...]}
        for item in data:
            img_key = item.get("image") or item.get("file_name") or item.get("img_name")
            if not img_key:
                continue
            for poly in item.get("polys", item.get("points", [])):
                bbox = normalize_poly(poly)
                if bbox:
                    gt[img_key].append(bbox)
    else:
        raise ValueError("Unsupported label file structure")

    return dict(gt)


def load_detector(model_dir: Path | None):
    """
    Placeholder for loading your detection model.
    Replace with actual model initialization logic.
    """
    # TODO: detector = YourDetector(model_dir)
    detector = None
    return detector


def run_detector(detector, image_path: Path) -> List[Box]:
    """
    Placeholder for running detection model inference.
    Replace with actual detector forward call.
    """
    # TODO: preds = detector.predict(image_path)
    # Expected return format: List[[x1, y1, x2, y2], ...]
    preds: List[Box] = []
    return preds


def evaluate(detector, image_dir: Path, gt: Dict[str, List[Box]], iou_thresh: float = 0.5):
    tp = fp = fn = 0
    iou_sum = 0.0
    iou_count = 0

    image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    for img_path in tqdm(image_paths, desc="Evaluating"):
        gt_boxes = gt.get(img_path.name, [])
        pred_boxes = run_detector(detector, img_path)

        matches = match_boxes(pred_boxes, gt_boxes)

        for a_idx, b_idx, iou in matches:
            if a_idx is not None and b_idx is not None:
                if iou >= iou_thresh:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
                iou_sum += iou
                iou_count += 1
            elif a_idx is None:
                fp += 1
            elif b_idx is None:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_iou = iou_sum / iou_count if iou_count else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "evaluated_images": len(image_paths),
    }


def main():
    parser = argparse.ArgumentParser(description="OCR detection benchmark")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(r"E:\datasets\CGN\test\test_p1"),
        help="Directory containing test images",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path(r"E:\datasets\CGN\test\test_p1\label.json"),
        help="Ground-truth label file (JSON)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to detection model weights/config (used in load_detector)",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP/FP")

    args = parser.parse_args()

    gt = load_ground_truth(args.label_file)
    print(f"Loaded GT for {len(gt)} images from {args.label_file}")

    detector = load_detector(args.model_dir)
    metrics = evaluate(detector, args.image_dir, gt, iou_thresh=args.iou_thresh)

    print("\n=== Detection Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


