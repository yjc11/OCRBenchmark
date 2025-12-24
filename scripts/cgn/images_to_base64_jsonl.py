import argparse
import base64
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def images_to_jsonl(input_dir: Path):
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"输入路径不是有效目录: {input_dir}")

    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())

    if not image_files:
        print("⚠️ 目录下未找到图片文件")
        return

    output_path = input_dir.with_suffix(".jsonl")

    total = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8-sig") as fout:
        for img_path in image_files:
            try:
                b64 = image_to_base64(img_path)
                record = {img_path.name: b64}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
            except Exception as e:
                skipped += 1
                print(f"❌ 跳过图片 {img_path.name}，原因: {e}")

    print("✅ 转换完成")
    print(f"📂 输入目录: {input_dir}")
    print(f"📄 输出文件: {output_path}")
    print(f"🖼️ 处理图片数: {total}")
    print(f"⚠️ 跳过图片数: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文件夹下的图片编码为 base64 并写入 JSONL")
    parser.add_argument("input_dir", type=str, help="包含图片的目录路径")

    args = parser.parse_args()
    images_to_jsonl(Path(args.input_dir))
