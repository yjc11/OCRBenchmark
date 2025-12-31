import json
from pathlib import Path


def pp2rocr(pp_label_path, rocr_label_path):
    pp_label_path = Path(pp_label_path)
    rocr_label_path = Path(rocr_label_path)
    rocr_label_path.parent.mkdir(parents=True, exist_ok=True)
    rocr_label_path.touch(exist_ok=True)
    results = {}
    with open(pp_label_path, "r", encoding="utf-8") as f:
        for line in f:
            img_name, label, score = line.strip().split('\t')
            results[Path(img_name).name] = {'value': [label], 'tags': [], 'confidence': score}
    with open(rocr_label_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    pp_label_path = "/workspace/datasets/test_p1_rec/infer/epoch7_results.txt"
    rocr_label_path = "/workspace/datasets/test_p1_rec/infer/epoch7_results_rocr.txt"
    pp2rocr(pp_label_path, rocr_label_path)
