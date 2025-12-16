import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scoring.metrics import precision_recall
from scripts.common import point8_to_box


def main():
    parser = argparse.ArgumentParser(description='Calculate detection precision and recall metrics')
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--threshold", type=float, default=0.8, help='Threshold for precision and recall')
    args = parser.parse_args()
    pred_dir = Path(args.pred_path)
    label_dir = Path(args.label_path)
    threshold = args.threshold
    overall_precision = 0
    overall_recall = 0
    label_file_list = list(label_dir.glob("**/*.json"))

    # 用于存储每个图片的结果
    results = []

    for label_file_path in tqdm(label_file_list):
        pred_path = pred_dir / label_file_path.stem / f"{label_file_path.stem}_res.json"
        if not pred_path.exists():
            continue
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_json = json.load(f)
            pred_polys = point8_to_box(np.array(pred_json['rec_polys']))
        with open(label_file_path, "r", encoding="utf-8") as f:
            label_json = json.load(f)
            label_polys = point8_to_box(np.array([label['points'] for label in label_json]))
        precision_recall_dict = precision_recall(pred_polys, label_polys, threshold=threshold, penalize_double=False)
        precision = precision_recall_dict['precision']
        recall = precision_recall_dict['recall']
        overall_precision += precision
        overall_recall += recall

        # 记录每个图片的结果
        results.append({'image_name': label_file_path.stem, 'precision': precision, 'recall': recall})

    if len(label_file_list) > 0:
        # 保存Excel文件到pred_dir的同级目录
        if results:
            # 计算总体平均值（基于实际处理成功的文件数）
            num_processed = len(results)
            overall_precision /= num_processed
            overall_recall /= num_processed
            print(f"overall precision: {overall_precision}, overall recall: {overall_recall}")

            df = pd.DataFrame(results)
            # 添加总体统计行
            df_summary = pd.DataFrame(
                [{'image_name': 'Overall', 'precision': overall_precision, 'recall': overall_recall}]
            )
            df = pd.concat([df, df_summary], ignore_index=True)

            # 保存到pred_dir的同级目录
            excel_path = pred_dir.parent / f"{pred_dir.name}_det_score.xlsx"
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"Results saved to: {excel_path}")
        else:
            print("No prediction files found to process.")
    else:
        print("No label files found in the specified directory.")


if __name__ == '__main__':
    main()
