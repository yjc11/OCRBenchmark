import json
import os

import pandas as pd

from .core.std_recog_score import compare


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return data


def compare_two_files(pred, label, mode='normal', exclude_tags=[], use_dp=False):
    assert mode in ['normal', 'baidu']
    statistics = {}
    statistics_rare = {}
    statistics_cn_tw = {}
    for sample_id in label:
        label_item = label[sample_id]
        label_value = label_item['value'][-1]
        scene = label_item['scene']
        if not scene:
            scene = ''
        else:
            scene = scene[-1]
        tags = label_item['tags']

        should_exclude = False
        for tag in exclude_tags:
            if tag in tags:
                should_exclude = True
        if should_exclude:
            continue

        try:
            pred_value = pred[sample_id]['value'][-1]
        except:
            # print('%s not found in submission file' % sample_id)
            pred_value = ''

        # baidu special treatment
        if (mode == 'baidu') and (pred_value == '' or pred_value == '##'):
            continue

        is_match_punc, cer_value = compare(pred_value, label_value, True, use_dp)
        is_match_depunc, _ = compare(pred_value, label_value, False, use_dp)

        if 'rare' in tags:
            if scene not in statistics_rare:
                statistics_rare[scene] = {'N': 0, 'N_punc': 0, 'N_depunc': 0, 'cer': 0, 'result': []}
            statistics_rare[scene]['N'] += 1
            statistics_rare[scene]['N_punc'] += is_match_punc
            statistics_rare[scene]['N_depunc'] += is_match_depunc
            statistics_rare[scene]['cer'] += cer_value
            statistics_rare[scene]['result'].append([label_value, pred_value])
        elif 'CN-TW' in tags:
            if scene not in statistics_cn_tw:
                statistics_cn_tw[scene] = {'N': 0, 'N_punc': 0, 'N_depunc': 0, 'cer': 0, 'result': []}
            statistics_cn_tw[scene]['N'] += 1
            statistics_cn_tw[scene]['N_punc'] += is_match_punc
            statistics_cn_tw[scene]['N_depunc'] += is_match_depunc
            statistics_cn_tw[scene]['cer'] += cer_value
            statistics_cn_tw[scene]['result'].append([label_value, pred_value])
        else:
            if scene not in statistics:
                statistics[scene] = {'N': 0, 'N_punc': 0, 'N_depunc': 0, 'cer': 0, 'result': []}
            statistics[scene]['N'] += 1
            statistics[scene]['N_punc'] += is_match_punc
            statistics[scene]['N_depunc'] += is_match_depunc
            statistics[scene]['cer'] += cer_value
            statistics[scene]['result'].append([label_value, pred_value])
    return statistics, statistics_rare, statistics_cn_tw


def pandas_table(statistics):
    data_rows = []
    total_count = 0
    total_correct_punc = 0
    total_correct_depunc = 0
    total_cer = 0
    total_gt_length = 0
    for scene_key in statistics:
        scene_count = statistics[scene_key]['N']
        scene_correct_punc = statistics[scene_key]['N_punc']
        scene_correct_depunc = statistics[scene_key]['N_depunc']
        scene_cer = statistics[scene_key]['cer']
        total_count += scene_count
        total_correct_punc += scene_correct_punc
        total_correct_depunc += scene_correct_depunc
        total_cer += scene_cer

        total_gt_length += sum((len(result_pair[0]) for result_pair in statistics[scene_key]['result']))
        row = [scene_key, scene_count, scene_correct_punc, scene_correct_depunc, scene_cer, total_gt_length]
        data_rows.append(row)

    data_rows.append(['总和', total_count, total_correct_punc, total_correct_depunc, total_cer, total_gt_length])
    df = pd.DataFrame(data_rows)
    df.columns = ['scene', 'count', 'correct_count_punc', 'correct_count_depunc', 'total_cer', 'gt_length']
    df['acc_punc'] = df['correct_count_punc'] / df['count']
    df['acc_depunc'] = df['correct_count_depunc'] / df['count']
    df['avg_cer'] = df['total_cer'] / df['count']
    df['whole_cer'] = df['total_cer'] / df['gt_length']
    return df


def score(submission_path, label_path, mode, exclude_tags=[], use_dp=False):
    label = load_json(label_path)
    pred = load_json(submission_path)
    statistics, statistics_rare, statistics_cn_tw = compare_two_files(pred, label, mode, exclude_tags, use_dp)
    df = pandas_table(statistics)
    df_rare = pandas_table(statistics_rare)
    df_cn_tw = pandas_table(statistics_cn_tw)

    if len(statistics) > 0:
        print('*' * 25)
        print('常见字')
        print(df)

    if len(statistics_rare) > 0:
        print('*' * 25)
        print('生僻字')
        print(df_rare)

    if len(statistics_cn_tw) > 0:
        print('*' * 25)
        print('繁体字')
        print(df_cn_tw)
    return df, df_rare, df_cn_tw
