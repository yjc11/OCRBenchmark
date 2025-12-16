# Initialize PaddleOCR instance
import shutil
from pathlib import Path

from paddleocr import PPStructureV3
from PIL import Image

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    device="gpu",
)

# Run OCR inference on a sample image
# test_dir = r".\data\sample_png"
# output_dir = r".\data\outputs\ppstructurev3_png"
test_dir = r".\data\sample_pdf"
output_dir = r".\data\outputs\ppstructurev3_pdf"
for img_path in Path(test_dir).glob("**/*.pdf"):
    _output_dir = Path(output_dir) / img_path.stem
    _output_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.predict(input=img_path.as_posix())
    for i, res in enumerate(result):
        shutil.copy(img_path, _output_dir)
        res.save_to_json(save_path=_output_dir)
        res.save_to_markdown(save_path=_output_dir)
