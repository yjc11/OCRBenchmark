import json


def trans_rrc_to_ppocr(rrc_label_path, output_path):
    """
    rrc format:
        img_name: {
            "value": [value],
            "tags": []
        }
    ppocr format:
        img_name\tvalue
        img_name1\tvalue2
    """
    with open(rrc_label_path, "r", encoding="utf-8") as f:
        rrc_label = json.load(f)
    with open(output_path, "w", encoding="utf-8") as f:
        for img_name, rrc_label_item in rrc_label.items():
            f.write(f"images/{img_name}\t{rrc_label_item['value'][0].strip()}\n")


if __name__ == "__main__":
    rrc_label_path = r"E:\datasets\CGN\test\test_p1_rec\label.json"
    output_path = r"E:\datasets\CGN\test\test_p1_rec\label_ppocr.txt"
    trans_rrc_to_ppocr(rrc_label_path, output_path)
