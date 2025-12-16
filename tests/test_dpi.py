# Initialize PaddleOCR instance
import os
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm
import logging

logging.getLogger("paddleocr").setLevel(logging.WARNING)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
)


# Run OCR inference on a sample image
idps = [72, 144, 200, 240]
for idp in idps:
    test_dir = Path(f"E:\datasets\CGN\processed\idp_{idp}")
    output_dir = Path(f"E:\datasets\CGN\processed\idp_{idp}_ppocrv5")
    for img_path in tqdm(list(Path(test_dir).iterdir())):
        result = ocr.predict(input=img_path.as_posix())
        _output_dir = output_dir / img_path.stem
        _output_dir.mkdir(parents=True, exist_ok=True)
        for i, res in enumerate(result):
            # res.print()
            res.save_to_img(_output_dir)
            res.save_to_json(_output_dir)
