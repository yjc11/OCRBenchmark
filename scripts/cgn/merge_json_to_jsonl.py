import argparse
import json
from pathlib import Path


def merge_json_to_jsonl(input_dir: Path):
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"输入路径不是有效目录: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print("⚠️ 目录下未找到 .json 文件")
        return

    output_path = input_dir.with_suffix(".jsonl")

    total_lines = 0
    skipped_files = 0

    with open(output_path, "w", encoding="utf-8-sig") as fout:
        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as fin:
                    data = json.load(fin)

                # 如果是 list，逐条写
                if isinstance(data, list):
                    for item in data:
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                        total_lines += 1
                else:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    total_lines += 1

            except Exception as e:
                skipped_files += 1
                print(f"❌ 跳过文件 {json_path.name}，原因: {e}")

    print(f"✅ 合并完成")
    print(f"📂 输入目录: {input_dir}")
    print(f"📄 输出文件: {output_path}")
    print(f"🧾 写入行数: {total_lines}")
    print(f"⚠️ 跳过文件数: {skipped_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将目录下多个 JSON 文件合并为一个 JSONL 文件")
    parser.add_argument("input_dir", type=str, help="包含多个 .json 文件的目录路径")

    args = parser.parse_args()
    merge_json_to_jsonl(Path(args.input_dir))
