# Initialize PaddleOCR instance
import json 
import os
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm

# disable paddleocr log
os.environ["PADDLE_LOG_LEVEL"] = "WARNING"

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
)

# Run OCR inference on a sample image
test_dir = r"E:\projects\bbox-label-tool-release-1.4.4\Images\B"
output_dir = r"E:\projects\bbox-label-tool-release-1.4.4\Labels\B"
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
for img_path in tqdm(Path(test_dir).glob("**/*.png")):
    result = ocr.predict(input=img_path.as_posix())
    output_dir.mkdir(parents=True, exist_ok=True)
    save_result = []
    for i, res in enumerate(result):
        polys = res['rec_polys']
        texts = res['rec_texts']
        for poly, text in zip(polys, texts):
            save_result.append({
                'value': text,
                'points': poly.tolist(),
                'category': 'text',
                'shape': 'polygon',
            })
    with open(output_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
        json.dump(save_result, f, ensure_ascii=False, indent=2)