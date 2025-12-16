import base64
import json
import os
import random
import sys
from difflib import Differ
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import gradio as gr
import pandas as pd
from PIL import Image

from scoring.core.std_recog_score import compare


def diff_to_html(diff):
    """将diff结果转换为HTML格式的高亮文本"""
    if not diff:
        return ""

    html = []
    for text, status in diff:
        if status == '+':
            # 添加的文本用红色显示
            html.append(f'<span style="color: red; font-weight: bold;">{text}</span>')
        elif status == '-':
            # 删除的文本用绿色显示
            html.append(f'<span style="color: green; text-decoration: line-through;">{text}</span>')
        else:
            # 未变化的文本正常显示
            html.append(text)

    return ''.join(html)


def image_to_base64(image_path):
    """将图片转换为 Base64 以便在 HTML 中显示"""
    w, h = Image.open(image_path).size
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    return f'''
        <div style="position: relative; display: inline-block;">
            <img src="data:image/png;base64,{base64_str}" width="{w}" height="{h}" 
                 onclick="this.parentElement.querySelector('.modal').style.display='block'" 
                 style="cursor: pointer;">
            <div class="modal" style="display: none; position: fixed; z-index: 1000; 
                 left: 0; top: 0; width: 100%; height: 100%; 
                 background-color: rgba(0,0,0,0.9); overflow: auto;"
                 onclick="this.style.display='none'">
                <img src="data:image/png;base64,{base64_str}" 
                     style="margin: auto; display: block; max-width: 90%; max-height: 90%; 
                     position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);"
                     onclick="event.stopPropagation()">
            </div>
        </div>
    '''


def diff_text(text1, text2):
    d = Differ()
    output = [(token[2:], token[0] if token[0] != ' ' else None) for token in d.compare(text2, text1)]
    return output


def analyze_results(
    label_path, pred_path, image_dir, only_diff=False, scene=None, max_samples=None, progress=gr.Progress()
):
    print('Scene:', scene)

    # 处理 max_samples 参数：None 或 0 表示不限制
    if max_samples is not None:
        try:
            max_samples = int(max_samples)
            if max_samples <= 0:
                max_samples = None
        except (ValueError, TypeError):
            max_samples = None

    with open(label_path, 'r', encoding='utf-8') as f:
        labels = json.loads(f.read())
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.loads(f.read())
    print(f"Loaded {len(labels)} labels and {len(preds)} predictions.")

    # 如果设置了最大样本数，在加载后立即采样
    if max_samples is not None and max_samples > 0 and len(labels) > max_samples:
        print(f"Sampling {max_samples} samples from {len(labels)} labels.")
        sampled_keys = random.sample(list(labels.keys()), k=max_samples)
        labels = {key: labels[key] for key in sampled_keys}
        print(f"After sampling: {len(labels)} labels to process.")

    results = []
    statatis = {'word_cnt': [], 'total_length': [], 'h': [], 'w/h': []}
    for img_name, info in progress.tqdm(labels.items(), desc="Processing images"):
        gt = info['value'][0]

        try:
            pred = preds[img_name]['value'][0]
        except:
            continue
        if only_diff:
            score_depunc, _ = compare(pred=pred, label=gt, with_punctuation=False)
            if score_depunc == 1:
                continue
        if scene and info['scene'][0] != scene:
            continue
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"File {img_path} does not exist.")
            continue

        pil_img = Image.open(img_path)
        w, h = pil_img.width, pil_img.height
        statatis['word_cnt'].append(len(gt.split()))
        statatis['total_length'].append(len(gt))
        statatis['h'].append(h)
        statatis['w/h'].append(w / h)

        diff = diff_text(pred, gt)

        results.append(
            {
                "image_path": os.path.basename(img_path),
                "image": image_to_base64(img_path),
                "diff": diff_to_html(diff) if diff else "",  # + 用红色表示，-用绿色表示
                "ground_truth": gt,
                "prediction": pred,
            }
        )

    print(f"Analyzed {len(results)} images.")

    df = pd.DataFrame(statatis)
    error_status = {
        'total_images': len(results),
        'total_length': df['total_length'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(),
        'word_cnt': df['word_cnt'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(),
        'h': df['h'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(),
        'w/h': df['w/h'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(),
    }
    error_status = json.dumps(error_status, ensure_ascii=False, indent=2)

    global cached_results
    cached_results = pd.DataFrame(results)
    # return cached_results
    return "Analysis completed. Ready to display results.", error_status


G_ORDER_INDEX = 0


def show_result(res, order=False):
    samples = res.to_dict('records')
    if order:
        global G_ORDER_INDEX
        # 按image_name排序
        samples.sort(key=lambda x: x['image_path'])
        # 取G_ORDER_INDEX: G_ORDER_INDEX+20
        t1 = pd.DataFrame(samples[G_ORDER_INDEX : G_ORDER_INDEX + 20])
        G_ORDER_INDEX += 20
        if G_ORDER_INDEX >= len(samples):
            G_ORDER_INDEX = 0
    else:
        t1 = pd.DataFrame(random.sample(samples, k=min(20, len(samples))))

    # info 为 html 格式，包含 image_name 和 diff, 每行一个
    info = ''
    image_names = t1['image_path'].to_list()
    diffs = t1['diff'].to_list()
    for image_name, diff in zip(image_names, diffs):
        info += f"<div>{image_name}: {diff}</div>"

    # 将t1的image_path列删除
    t1 = t1.drop(columns=['image_path'])
    return t1, info


def create_interface():
    global cached_results
    with gr.Blocks() as demo:
        gr.Markdown("# 模型识别错例分析工具")

        with gr.Row():
            pred_path = gr.Textbox(
                label="模型预估结果",
                value=r'E:\datasets\CGN\processed\idp_144_ppocrv5_cropped/ppocrv5_predictions.json',
            )
            val_dir = gr.Textbox(
                label="验证集图片目录", value=r'E:\datasets\CGN\processed\idp_144_ppocrv5_cropped\images'
            )
            label_path = gr.Textbox(
                label="验证集标签文件路径",
                value=r'E:\datasets\CGN\processed\idp_144_ppocrv5_cropped/phocr_predictions.json',
            )
            scene = gr.Textbox(label='Scene', value='doc')

        with gr.Row():
            only_diff = gr.Checkbox(label="只显示有差异的样本")
            max_samples = gr.Number(label="最大样本数（0或空表示不限制）", value=0, precision=0)

        # 加一个文本框，显示错误的统计信息
        error_status = gr.Textbox(label="错误统计信息", value='')

        with gr.Row():
            with gr.Column():
                analyze_btn = gr.Button("开始分析")
                text_output = gr.Textbox(label="分析进度")
                show_btn = gr.Button("开始展示")
                show_btn2 = gr.Button("顺序展示")

        output_table = gr.DataFrame(
            label="分析结果(随机展示 20个)",
            headers=["图像", "diff", "ground_truth", "prediction"],
            datatype=["html", "html", "str", "str"],
            render=True,
        )

        info_status = gr.HTML(label="文本编辑信息", value='')

        analyze_btn.click(
            fn=analyze_results,
            inputs=[label_path, pred_path, val_dir, only_diff, scene, max_samples],
            outputs=[text_output, error_status],
        )

        show_btn.click(
            fn=lambda: show_result(cached_results),
            inputs=[],
            outputs=[output_table, info_status],
        )

        show_btn2.click(
            fn=lambda: show_result(cached_results, order=True),
            inputs=[],
            outputs=[output_table, info_status],
        )

    return demo


def analyze_and_save(label_path, pred_path, image_dir, only_diff=False):
    df = analyze_results(label_path, pred_path, image_dir, only_diff)
    output_path = os.path.join(os.path.dirname(pred_path), "analysis_results.json")
    df.to_json(output_path, orient="records", lines=True)
    print(f"Results saved to {output_path}")


def trans_txt_label_to_json(txt_label_path=None, json_label_path=None):
    json_label_path = './output/datasets/ru_doc_test_v1/labels.json'
    txt_label_path = './output/datasets/ru_doc_test_v1/labels.txt'

    json_label = {}
    with open(txt_label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        img_name, label = line.strip().split('\t', 1)
        img_name = os.path.basename(img_name)
        json_label[img_name] = {'value': [label], 'scene': ['doc']}

    with open(json_label_path, 'w') as f:
        json.dump(json_label, f, ensure_ascii=False)


def trans_api_result_to_json(txt_label_path=None, json_label_path=None):
    json_label_path = './output/datasets/th_doc_test_v1/ali_result.json'
    api_result_path = './output/datasets/th_doc_test_v1/labels_ali.jsonl'

    json_label = {}
    with open(api_result_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        content = json.loads(line)
        img_name, label = content['image_name'], content['texts'][0]
        img_name = os.path.basename(img_name)
        json_label[img_name] = {'value': [label], 'scene': ['doc']}

    with open(json_label_path, 'w') as f:
        json.dump(json_label, f, ensure_ascii=False)


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="localhost", server_port=8800)
