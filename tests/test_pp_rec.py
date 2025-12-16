import json
import os
import tempfile
from pathlib import Path

from paddleocr import TextRecognition
from tqdm import tqdm

# disable paddleocr log
os.environ["PADDLE_LOG_LEVEL"] = "WARNING"

# Initialize TextRecognition model
model = TextRecognition(model_name="PP-OCRv5_server_rec")

# Configuration
test_dir = r"E:\datasets\CGN\test\test_p1_rec\images"  # 修改为你的图片目录
output_json_path = r"E:\datasets\CGN\test\pp_rec_results.json"  # 输出 JSON 文件路径

# 存储所有图片的识别结果
results = {}

# 遍历图片目录进行识别
img_paths = list(Path(test_dir).glob("**/*.png"))
if not img_paths:
    img_paths = list(Path(test_dir).glob("**/*.jpg"))
if not img_paths:
    img_paths = list(Path(test_dir).glob("**/*.jpeg"))

print(f"Found {len(img_paths)} images to process")

for img_path in tqdm(img_paths, desc="Processing images"):
    try:
        # 运行 OCR 识别
        output = model.predict(input=img_path.as_posix(), batch_size=1)
        rec_text = output[0]['rec_text']
        results[img_path.name] = {"value": [rec_text], "tags": []}
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        # 即使出错也记录，value 为空列表
        results[img_path.name] = {"value": [], "tags": []}

# 保存结果到 JSON 文件
output_path = Path(output_json_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_path}")
print(f"Total images processed: {len(results)}")
