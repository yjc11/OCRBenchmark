"""
使用说明：在命令行运行 python search.py --image-dir <图片目录> --label-dir <标注目录>，脚本会根据图片名在标注目录中搜索对应的JSON文件并复制到输出目录。
"""

import argparse
import shutil
from pathlib import Path


def search_label(image_dir: Path, label_dir: Path):
    """
    根据图片目录下的所有图片名，在标注目录下搜索对应的标注文件，如果存在，则复制到图片统计目录下{图片文件夹目录}_label文件夹下
    """
    output_dir = image_dir.parent / f"{image_dir.name}_label"
    output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in image_dir.glob("**/*.png"):
        img_name = img_path.stem
        # 搜索labeldir下所有json文件，如果文件名包含img_name，则复制到output_dir下
        for label_file in label_dir.glob("**/*.json"):
            if img_name in label_file.stem:
                shutil.copy(label_file, output_dir / label_file.name)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据图片名从标注目录查找对应json并复制到输出目录")
    parser.add_argument("--image-dir", type=Path, required=True, help="包含待查找图片的目录")
    parser.add_argument("--label-dir", type=Path, required=True, help="包含全部json标注的目录")
    args = parser.parse_args()

    search_label(Path(args.image_dir), Path(args.label_dir))
    print("Done")
