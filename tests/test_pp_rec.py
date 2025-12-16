import argparse
import json
import os
from pathlib import Path

from paddleocr import TextRecognition
from tqdm import tqdm

# disable paddleocr log
os.environ["PADDLE_LOG_LEVEL"] = "WARNING"


def main():
    parser = argparse.ArgumentParser(description='Run PPOCR text recognition on images')
    parser.add_argument("--input_dir", type=str, required=True, help='Input directory containing images')
    parser.add_argument("--output_path", type=str, required=True, help='Output JSON file path')
    parser.add_argument(
        "--model_name",
        type=str,
        default="PP-OCRv5_server_rec",
        help='Model name for TextRecognition (default: PP-OCRv5_server_rec)',
    )
    parser.add_argument("--batch_size", type=int, default=1, help='Batch size for prediction (default: 1)')
    args = parser.parse_args()

    # Initialize TextRecognition model
    model = TextRecognition(model_name=args.model_name)

    # Configuration
    test_dir = Path(args.input_dir)
    output_json_path = Path(args.output_path)

    if not test_dir.exists():
        print(f"Error: Input directory does not exist: {test_dir}")
        return

    # 存储所有图片的识别结果
    results = {}

    # 遍历图片目录进行识别
    img_paths = list(test_dir.glob("**/*.png"))
    if not img_paths:
        img_paths = list(test_dir.glob("**/*.jpg"))
    if not img_paths:
        img_paths = list(test_dir.glob("**/*.jpeg"))

    if not img_paths:
        print(f"Error: No images found in {test_dir}")
        return

    print(f"Found {len(img_paths)} images to process")

    for img_path in tqdm(img_paths, desc="Processing images"):
        try:
            # 运行 OCR 识别
            output = model.predict(input=img_path.as_posix(), batch_size=args.batch_size)
            rec_text = output[0]['rec_text']
            results[img_path.name] = {"value": [rec_text], "tags": []}
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # 即使出错也记录，value 为空列表
            results[img_path.name] = {"value": [], "tags": []}

    # 保存结果到 JSON 文件
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_json_path}")
    print(f"Total images processed: {len(results)}")


if __name__ == '__main__':
    main()
