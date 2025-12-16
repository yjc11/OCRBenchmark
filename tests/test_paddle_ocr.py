# Initialize PaddleOCR instance
import os
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image

# disable paddleocr log
os.environ["PADDLE_LOG_LEVEL"] = "WARNING"

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
)

# Run OCR inference on a sample image
test_dir = r"E:\datasets\CGN\test\test_p1"
output_dir = r"E:\datasets\CGN\test\test_p1_ppocrv5"
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
for img_path in Path(test_dir).glob("**/*.png"):
    result = ocr.predict(input=img_path.as_posix())
    _output_dir = output_dir / img_path.stem
    _output_dir.mkdir(parents=True, exist_ok=True)
    for i, res in enumerate(result):
        # res.print()
        # res.save_to_img(_output_dir)
        res.save_to_json(_output_dir)