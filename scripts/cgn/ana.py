import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 添加父目录到路径，以便导入common模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import perspective_transform

data_dir = Path(r"E:\datasets\CGN\from_zh\json_batch_run")
image_dir = Path(r"E:\datasets\CGN\from_zh\png")
output_dir = Path(r"E:\datasets\CGN\from_zh\cropped_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Excel文件路径，包含允许的字段名称列表
field_info_excel = Path(r"E:\projects\ocr_benchmark\scripts\cgn\字段信息.xlsx")

# 配置选项：是否使用多进程（False为单进程模式，True为多进程模式）
USE_MULTIPROCESSING = True


def load_allowed_field_names(excel_path: Path) -> set:
    """从Excel文件中读取允许的字段名称列表"""
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        # 获取"字段名称"列的所有值，去除空值并转换为集合
        if "code" in df.columns:
            field_names = df["code"].dropna().astype(str).unique()
            return set(field_names)
        else:
            print(f"Warning: 'code' column not found in {excel_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return set()
    except Exception as e:
        print(f"Error loading field names from {excel_path}: {e}")
        return set()


def process_single_json(args):
    """处理单个JSON文件的函数"""
    json_file, image_dir, output_dir, allowed_field_names = args

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 获取对应的图片文件名基础部分（假设json文件名和图片文件名相同，只是扩展名不同）
        image_stem = json_file.stem

        # 获取page_data
        page_data = data.get("page_data", [])
        if not page_data:
            return {"warnings": [f"No page_data found in {json_file}"]}

        warnings = []

        # 处理每个页面
        for page_idx, page in enumerate(page_data):
            # 获取page_num，用于构建图片文件名
            page_num = page.get("page_num", page_idx)

            # 根据page_num查找对应的图片文件
            image_path = None
            # 尝试常见的图片扩展名，文件名格式为: {image_stem}_page_{page_num}.{ext}
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
                potential_path = image_dir / f"{image_stem}_page_{page_num}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break

            # 如果找不到带page_num的文件，尝试不带page_num的文件（兼容性）
            if image_path is None:
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
                    potential_path = image_dir / f"{image_stem}{ext}"
                    if potential_path.exists():
                        image_path = potential_path
                        break

            if image_path is None:
                warnings.append(f"Image file not found for {json_file}, page_num={page_num}")
                continue

            # 读取图片
            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)
            img_width, img_height = img.size

            layout_list = page.get("layout_list", [])

            # 处理每个layout
            for layout_idx, layout in enumerate(layout_list):
                # 获取字段值用于分文件夹（使用layout_type或layout_label）
                field_value = (
                    layout.get("layout_type") or layout.get("layout_label") or layout.get("layout_name") or "unknown"
                )

                # 创建对应的文件夹
                field_dir = output_dir / str(field_value)
                field_dir.mkdir(parents=True, exist_ok=True)

                # 获取layout_location（归一化坐标）
                layout_location = layout.get("layout_location", [])
                if layout_location and len(layout_location) >= 4:
                    # 将归一化坐标转换为像素坐标
                    # layout_location格式: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]，坐标范围0-1
                    pts_normalized = layout_location
                    pts_pixel = [(float(pt[0]) * img_width, float(pt[1]) * img_height) for pt in pts_normalized]

                    # 使用透视变换裁剪图片
                    crop_img_np = perspective_transform(img_np, pts_pixel)
                    crop_img = Image.fromarray(crop_img_np)

                    # 生成保存文件名（使用原图名+页面号+layout索引）
                    crop_img_name = f"{image_stem}_page{page_num}_l{layout_idx}.png"
                    crop_img_path = field_dir / crop_img_name
                    crop_img.save(crop_img_path)

                # 处理ocr_result_list中的每个OCR结果
                ocr_result_list = layout.get("ocr_result_list", [])
                for ocr_idx, ocr_result in enumerate(ocr_result_list):
                    # OCR结果也按相同的字段分文件夹
                    ocr_field_dir = field_dir / "ocr_texts"
                    ocr_field_dir.mkdir(parents=True, exist_ok=True)

                    # 获取OCR location（归一化坐标，格式可能是[x1, y1, x2, y2]）
                    ocr_location = ocr_result.get("location", [])
                    if ocr_location and len(ocr_location) >= 4:
                        # 将归一化坐标转换为像素坐标
                        x1 = float(ocr_location[0]) * img_width
                        y1 = float(ocr_location[1]) * img_height
                        x2 = float(ocr_location[2]) * img_width
                        y2 = float(ocr_location[3]) * img_height

                        # 转换为4个点的格式（矩形）
                        pts_pixel = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                        # 使用透视变换裁剪图片
                        crop_img_np = perspective_transform(img_np, pts_pixel)
                        crop_img = Image.fromarray(crop_img_np)

                        # 获取文本内容用于文件名
                        text = ocr_result.get("text", "")
                        # 清理文本，移除不适合作为文件名的字符
                        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).strip()
                        safe_text = safe_text[:20]  # 限制长度
                        if not safe_text:
                            safe_text = "text"

                        # 生成保存文件名
                        crop_img_name = f"{image_stem}_page{page_num}_l{layout_idx}_ocr{ocr_idx}_{safe_text}.png"
                        crop_img_path = ocr_field_dir / crop_img_name
                        crop_img.save(crop_img_path)

                # 处理fields_data_list中的小碎图（二维列表结构）
                fields_data_list = layout.get("fields_data_list", [])
                for row_idx, field_row in enumerate(fields_data_list):
                    # 处理每一行中的每个字段
                    for field_idx, field_data in enumerate(field_row):
                        # 获取field_name用于分文件夹
                        field_name = field_data.get("field_code") or field_data.get("name") or "unknown"

                        # 过滤：只处理在允许列表中的字段
                        if allowed_field_names and field_name not in allowed_field_names:
                            continue

                        # 创建对应的文件夹（按field_name分文件夹）
                        field_name_dir = output_dir / "fields" / str(field_name)
                        field_name_dir.mkdir(parents=True, exist_ok=True)

                        # 获取字段的位置信息（使用value_location）
                        value_location = field_data.get("value_location", [])

                        if value_location and len(value_location) >= 4:
                            # 格式: [x1, y1, x2, y2] (归一化坐标)
                            x1 = float(value_location[0]) * img_width
                            y1 = float(value_location[1]) * img_height
                            x2 = float(value_location[2]) * img_width
                            y2 = float(value_location[3]) * img_height

                            # 转换为4个点的格式（矩形）
                            pts_pixel = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                            # 使用透视变换裁剪图片
                            crop_img_np = perspective_transform(img_np, pts_pixel)
                            crop_img = Image.fromarray(crop_img_np)

                            # 生成保存文件名（包含行索引和字段索引）
                            crop_img_name = (
                                f"{image_stem}_page{page_num}_l{layout_idx}_row{row_idx}_field{field_idx}.png"
                            )
                            crop_img_path = field_name_dir / crop_img_name
                            crop_img.save(crop_img_path)

        return {"warnings": warnings}

    except Exception as e:
        import traceback

        error_msg = f"Error processing {json_file}: {e}"
        traceback.print_exc()
        return {"error": error_msg, "warnings": []}


if __name__ == "__main__":
    # 读取允许的字段名称列表
    allowed_field_names = load_allowed_field_names(field_info_excel)
    if allowed_field_names:
        print(f"Loaded {len(allowed_field_names)} allowed field names from {field_info_excel}")
        print(f"Allowed fields: {sorted(allowed_field_names)}")
    else:
        print(f"Warning: No allowed field names loaded. All fields will be saved.")
        print(f"Please check if the Excel file exists and contains '字段名称' column.")

    # 获取所有JSON文件
    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        exit(1)

    # 准备参数列表（包含允许的字段名称列表）
    args_list = [(json_file, image_dir, output_dir, allowed_field_names) for json_file in json_files]

    results = []

    if USE_MULTIPROCESSING:
        # 使用多进程处理
        num_workers = cpu_count()
        print(f"Using {num_workers} processes to process {len(json_files)} JSON files...")

        # 处理所有JSON文件
        with Pool(processes=num_workers) as pool:
            # 使用 imap 以便使用 tqdm 显示进度
            results = list(
                tqdm(pool.imap(process_single_json, args_list), total=len(args_list), desc="Processing JSON files")
            )
    else:
        # 使用单进程处理
        print(f"Using single process to process {len(json_files)} JSON files...")
        results = [process_single_json(args) for args in tqdm(args_list, desc="Processing JSON files")]

    # 统计警告和错误
    total_warnings = 0
    total_errors = 0
    for result in results:
        if "error" in result:
            total_errors += 1
        if "warnings" in result:
            total_warnings += len(result["warnings"])

    print(f"\nProcessing completed!")
    print(f"Total files processed: {len(json_files)}")
    print(f"Errors: {total_errors}")
    print(f"Warnings: {total_warnings}")
    print(f"All cropped images saved to: {output_dir}")
