import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# from scoring.rec_metric import calculate_cer
from scoring.score import score


def main():
    parser = argparse.ArgumentParser(description='Calculate detection precision and recall metrics')
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    pred_dir = Path(args.pred_path)
    label_dir = Path(args.label_path)
    df, df_rare, df_CN_TW = score(pred_dir, label_dir, mode='normal')
    pd.DataFrame(df).to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
