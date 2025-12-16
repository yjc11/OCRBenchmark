import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scoring.score_rec import calculate_cer


def main():
    parser = argparse.ArgumentParser(description='Calculate detection precision and recall metrics')
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--label_path", type=str)
    args = parser.parse_args()
    pred_dir = Path(args.pred_path)
    label_dir = Path(args.label_path)
    overall_cer = 0
    with open(pred_dir, "r", encoding="utf-8") as f:
        pred_json = json.load(f)
    with open(label_dir, "r", encoding="utf-8") as f:
        label_json = json.load(f)
    preds = []
    labels = []
    for pred_name, pred_value in pred_json.items():
        preds.append(pred_value['value'][0])
        labels.append(label_json[pred_name]['value'][0])
    cer, _ = calculate_cer(preds, labels)
    print(f"Overall CER: {1 - cer}")


if __name__ == "__main__":
    main()
