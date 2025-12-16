# import re
import Levenshtein
import pandas as pd
import regex as re


def reduce_underscores(text: str) -> str:
    return re.sub(r'_+', '_', text)


def normalize_text(text):
    text = (
        text.replace('\\t', '')
        .replace('\\n', '')
        .replace('\t', '')
        .replace('\n', '')
        .replace('/t', '')
        .replace('/n', '')
    )
    text = text.replace('[UNK]', '').replace('[unk]', '')
    text = ' '.join(text.split())
    text = reduce_underscores(text)
    return text


def clean_string(input_string):
    # Use regex to keep Chinese characters, English letters and numbers
    input_string = (
        input_string.replace('\\t', '')
        .replace('\\n', '')
        .replace('\t', '')
        .replace('\n', '')
        .replace('/t', '')
        .replace('/n', '')
        .replace('_', '')
        .replace(' ', '')
    )
    input_string = ''.join(input_string.split())
    cleaned_string = re.sub(r'[\p{S}\p{P}]', '', input_string)
    return cleaned_string


def calculate_cer(preds, gts, depunctuation=False):
    edit_dist_list = []
    gts_cnt = []
    cnt = 0
    for pred, gt in zip(preds, gts):
        pred = normalize_text(pred)
        gt = normalize_text(gt)
        if depunctuation:
            pred = clean_string(pred)
            gt = clean_string(gt)
        if len(gt) == 0:
            continue
        edit_dist = Levenshtein.distance(pred, gt)
        if edit_dist != 0:
            cnt += 1
        edit_dist_list.append(edit_dist)
        gts_cnt.append(len(gt))
    return sum(edit_dist_list) / sum(gts_cnt), sum(gts_cnt)
