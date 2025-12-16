import hashlib
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from common import perspective_transform
from PIL import Image
from tqdm import tqdm


def process_single_image1(args):
    """处理单个图像目录的函数, 结果文件为paddleocr的格式"""
    image_dir, ori_image_dir, image_output_dir = args

    if not image_dir.is_dir():
        return {}

    image_name = image_dir.name
    image_path = ori_image_dir / f"{image_name}.png"
    json_path = image_dir / f"{image_name}_res.json"

    if not image_path.exists() or not json_path.exists():
        print(f"image_path or json_path not found: {image_path} or {json_path}")
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        dt_polys = json_data.get("dt_polys", [])
        rec_texts = json_data.get("rec_texts", [])

        crop_record = {}

        for idx, pts in enumerate(dt_polys):
            # 封装 pts
            pts_tuple = [tuple(pt) for pt in pts]  # [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            crop_img_np = perspective_transform(img_np, pts_tuple)
            crop_img = Image.fromarray(crop_img_np)

            # 生成唯一图片md5用于保存
            md5 = hashlib.md5()
            md5.update(np.array(crop_img).tobytes())
            crop_img_name = f"{md5.hexdigest()}.png"
            crop_img.save(image_output_dir / crop_img_name)

            # 获取对应的预测文字，tag留空
            pred_value = rec_texts[idx] if idx < len(rec_texts) else ""
            crop_record[crop_img_name] = {'value': [pred_value], 'tags': [], 'from_image': image_name}

        return crop_record
    except Exception as e:
        print(f"Error processing {image_dir}: {e}")
        return {}


def process_single_image2(args):
    """
    图片和标注文件在同一个目录下，图片名为标注文件的文件名，标注文件为json格式，格式为：
    [
      {
        "category": "text",
        "value": "性能规范",
        "shape": "polygon",
        "points": [
          [
            4536.0,
            96.0
          ],
          [
            4814.0,
            96.0
          ],
          [
            4814.0,
            165.0
          ],
          [
            4536.0,
            165.0
          ]
        ]
      },
      {
        "category": "text",
        "value": "公称压力",
        "shape": "polygon",
        "points": [
          [
            4074.0,
            189.0
          ],
          [
            4243.0,
            189.0
          ],
          [
            4243.0,
            240.0
          ],
          [
            4074.0,
            240.0
          ]
        ]
      },
      ...
      ]
    """
    label_tool_dir, output_dir = args
    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = output_dir / "label.json"
    json_output_path.touch(exist_ok=True)

    if not label_tool_dir.is_dir():
        return {}

    label_tool_files = [f for f in label_tool_dir.iterdir() if f.is_file() and f.suffix == ".json"]

    all_crop_record = {}

    for label_tool_file in tqdm(label_tool_files, desc="Processing label tool files"):
        try:
            with open(label_tool_file, "r", encoding="utf-8") as f:
                label_tool_data = json.load(f)

            # 找到对应的图片文件（图片名为标注文件的文件名）
            image_stem = label_tool_file.stem
            image_path = None

            # 尝试常见的图片扩展名
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
                potential_path = label_tool_dir / f"{image_stem}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break

            if image_path is None:
                print(f"Image file not found for {label_tool_file}")
                continue

            # 读取图片
            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)

            # 处理每个标注项
            for item in label_tool_data:
                if item.get("shape") != "polygon":
                    continue

                points = item.get("points", [])
                value = item.get("value", "")

                if len(points) < 4:
                    continue

                # 将 points 转换为 tuple 格式: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                pts_tuple = [tuple(pt) for pt in points]

                # 使用透视变换裁剪图片
                crop_img_np = perspective_transform(img_np, pts_tuple)
                crop_img = Image.fromarray(crop_img_np)

                # 生成唯一图片md5用于保存
                md5 = hashlib.md5()
                md5.update(np.array(crop_img).tobytes())
                crop_img_name = f"{md5.hexdigest()}.png"
                crop_img.save(image_output_dir / crop_img_name)

                # 记录裁剪信息
                all_crop_record[crop_img_name] = {'value': [value], 'tags': [], 'from_image': image_stem}

        except Exception as e:
            print(f"Error processing {label_tool_file}: {e}")
            continue

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_crop_record, f, ensure_ascii=False, indent=2)

    return all_crop_record


def run_process_single_image1():
    data_dir = Path(r"E:\datasets\CGN\processed\idp_144_ppocrv5")
    ori_image_dir = Path(r"E:\datasets\CGN\processed\idp_144")
    output_dir = Path(r"E:\datasets\CGN\processed\idp_144_ppocrv5_cropped")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有需要处理的图像目录
    image_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    # 准备参数列表
    args_list = [(image_dir, ori_image_dir, image_output_dir) for image_dir in image_dirs]

    # 使用多进程处理
    num_workers = cpu_count()
    print(f"Using {num_workers} processes...")

    all_crop_record = {}
    with Pool(processes=num_workers) as pool:
        # 使用 imap 以便使用 tqdm 显示进度
        results = list(
            tqdm(pool.imap(process_single_image1, args_list), total=len(args_list), desc="Processing images")
        )

    # 合并所有结果
    for crop_record in results:
        all_crop_record.update(crop_record)

    # 保存全部记录信息到一个json中
    output_json_path = output_dir / "all_crop_record.json"
    with open(output_json_path, "w", encoding="utf-8") as fout:
        json.dump(all_crop_record, fout, ensure_ascii=False, indent=2)

    print(f"Processed {len(all_crop_record)} cropped images.")


if __name__ == '__main__':
    data_dir = Path(r'E:\datasets\CGN\test\test_p1')
    output_dir = Path(r'E:\datasets\CGN\test\test_p1_rec')
    output_dir.mkdir(parents=True, exist_ok=True)
    process_single_image2((data_dir, output_dir))
    pass
