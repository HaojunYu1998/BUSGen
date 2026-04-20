"""
Build KDE prior files for BoxSampler from a dataset with bounding box annotations.

Input: A JSON file where each entry has:
    {
        "pathology": "benign" or "malignant",
        "lesion_box": {"1": [x1, y1, x2, y2], ...}   # normalized [0, 1]
    }

Output: 6 KDE pkl files in ModelFiles/:
    {benign,malignant}_{center,aspect_ratio,area}_kde.pkl

Usage:
    python build_kde_priors.py --json_file DatasetFiles/BUSI_DevAug.json --output_dir ModelFiles
"""

import os
import json
import argparse
import joblib
import numpy as np
from sklearn.neighbors import KernelDensity


def extract_box_features(data):
    """Extract center, aspect_ratio, area from bounding boxes, grouped by pathology."""
    features = {"benign": [], "malignant": []}

    for item in data:
        pathology = item["pathology"]
        if pathology not in features:
            continue

        boxes = item["lesion_box"]
        if isinstance(boxes, dict):
            boxes = list(boxes.values())
        elif isinstance(boxes, list) and len(boxes) == 4 and isinstance(boxes[0], (int, float)):
            boxes = [boxes]

        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = w * h
            aspect_ratio = h / w

            features[pathology].append({
                "center": [cx, cy],
                "aspect_ratio": [aspect_ratio],
                "area": [area],
            })

    return features


def fit_kde(data, bandwidth=0.001, kernel="gaussian"):
    """Fit a KDE model on the given data."""
    data = np.array(data)
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(data)
    return kde


def main():
    parser = argparse.ArgumentParser(description="Build KDE priors for BoxSampler")
    parser.add_argument("--json_file", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="ModelFiles", help="Output directory for KDE pkl files")
    parser.add_argument("--bandwidth", type=float, default=0.001, help="KDE bandwidth (default: 0.001)")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    # Filter valid entries
    data = [x for x in data if x.get("is_valid", True)]
    print(f"Loaded {len(data)} valid entries from {args.json_file}")

    features = extract_box_features(data)
    os.makedirs(args.output_dir, exist_ok=True)

    for pathology in ["benign", "malignant"]:
        items = features[pathology]
        if not items:
            print(f"WARNING: No {pathology} samples found, skipping.")
            continue

        print(f"\n{pathology}: {len(items)} bounding boxes")

        for attr in ["center", "aspect_ratio", "area"]:
            attr_data = [item[attr] for item in items]
            kde = fit_kde(attr_data, bandwidth=args.bandwidth)

            output_path = os.path.join(args.output_dir, f"{pathology}_{attr}_kde.pkl")
            joblib.dump(kde, output_path)

            arr = np.array(attr_data)
            print(f"  {attr}: shape={arr.shape}, "
                  f"mean={arr.mean(axis=0).round(4)}, "
                  f"std={arr.std(axis=0).round(4)}, "
                  f"range=[{arr.min(axis=0).round(4)}, {arr.max(axis=0).round(4)}]")
            print(f"    -> saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
