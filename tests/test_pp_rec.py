import argparse
import json
import os
import tempfile
from pathlib import Path

from paddleocr import TextRecognition
from tqdm import tqdm

# disable paddleocr log
os.environ["PADDLE_LOG_LEVEL"] = "WARNING"


def main():
    parser = argparse.ArgumentParser(description='Test PP-OCRv5')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--output_json_path', type=str)
    args = parser.parse_args()
    model_path = args.model_path
    test_dir = args.test_dir
    output_json_path = args.output_json_path
    model = TextRecognition(model_dir=model_path)

    # Configuration
    results = {}
    for img_path in tqdm(list(Path(test_dir).glob("**/*.png")), desc="Processing images"):
        try:
            output = model.predict(input=str(img_path), batch_size=1)
            rec_text = output[0]['rec_text']
            results[img_path.name] = {"value": [rec_text], "tags": [], 'scene': []}
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results[img_path.name] = {"value": [], "tags": [], 'scene': []}

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total images processed: {len(results)}")


if __name__ == "__main__":
    main()
