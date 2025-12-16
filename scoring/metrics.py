from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat

import numpy as np


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_iou(box1, box2, box1_only=False):
    intersection = intersection_area(box1, box2)
    union = box_area(box1)
    if not box1_only:
        union += box_area(box2) - intersection

    if union == 0:
        return 0
    return intersection / union


def match_boxes(preds, references):
    num_actual = len(references)
    num_predicted = len(preds)

    iou_matrix = np.zeros((num_actual, num_predicted))
    for i, actual in enumerate(references):
        for j, pred in enumerate(preds):
            iou_matrix[i, j] = calculate_iou(actual, pred, box1_only=True)

    sorted_indices = np.argsort(iou_matrix, axis=None)[::-1]
    sorted_ious = iou_matrix.flatten()[sorted_indices]
    actual_indices, predicted_indices = np.unravel_index(sorted_indices, iou_matrix.shape)

    assigned_actual = set()
    assigned_pred = set()

    matches = []
    for idx, iou in zip(zip(actual_indices, predicted_indices), sorted_ious):
        i, j = idx
        if i not in assigned_actual and j not in assigned_pred:
            iou_val = iou_matrix[i, j]
            if iou_val > 0.95:  # Account for rounding on box edges
                iou_val = 1.0
            matches.append((i, j, iou_val))
            assigned_actual.add(i)
            assigned_pred.add(j)

    unassigned_actual = set(range(num_actual)) - assigned_actual
    unassigned_pred = set(range(num_predicted)) - assigned_pred
    matches.extend([(i, None, -1.0) for i in unassigned_actual])
    matches.extend([(None, j, 0.0) for j in unassigned_pred])

    return matches


def penalized_iou_score(preds, references):
    matches = match_boxes(preds, references)
    iou = sum([match[2] for match in matches]) / len(matches)
    return iou


def intersection_pixels(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return set()

    x_left, x_right = int(x_left), int(x_right)
    y_top, y_bottom = int(y_top), int(y_bottom)

    coords = np.meshgrid(np.arange(x_left, x_right), np.arange(y_top, y_bottom))
    pixels = set(zip(coords[0].flat, coords[1].flat))

    return pixels


def calculate_coverage(box, other_boxes, penalize_double=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    # find total coverage of the box
    covered_pixels = set()
    double_coverage = list()
    for other_box in other_boxes:
        ia = intersection_pixels(box, other_box)
        double_coverage.append(list(covered_pixels.intersection(ia)))
        covered_pixels = covered_pixels.union(ia)

    # Penalize double coverage - having multiple bboxes overlapping the same pixels
    double_coverage_penalty = len(double_coverage)
    if not penalize_double:
        double_coverage_penalty = 0
    covered_pixels_count = max(0, len(covered_pixels) - double_coverage_penalty)
    return covered_pixels_count / box_area


def intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def calculate_coverage_fast(box, other_boxes, penalize_double=False):
    """
    快速计算box被other_boxes覆盖的比例。
    该函数为每个box计算所有other_boxes与其的交集面积和，
    再与box的面积相除，得到coverage比例（最多为1.0）。
    注：penalize_double参数在本实现中不生效，只为接口兼容。
    
    Args:
        box: 单个待评估的框（如[x1, y1, x2, y2]）。
        other_boxes: 用于与box计算重叠的其他框，shape为(N, 4)。
        penalize_double: 保留接口，无实际作用。

    Returns:
        box被所有other_boxes覆盖的面积占比（0~1）。
    """
    box = np.array(box)
    other_boxes = np.array(other_boxes)

    # 计算box的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    # 分别计算每个other_box与box的交集区域的左上和右下坐标
    x_left = np.maximum(box[0], other_boxes[:, 0])
    y_top = np.maximum(box[1], other_boxes[:, 1])
    x_right = np.minimum(box[2], other_boxes[:, 2])
    y_bottom = np.minimum(box[3], other_boxes[:, 3])

    # 交集区域的宽和高，小于0则取0（表示无重叠）
    widths = np.maximum(0, x_right - x_left)
    heights = np.maximum(0, y_bottom - y_top)
    # 各交集面积
    intersect_areas = widths * heights

    # 累加所有交集面积，得到总的被覆盖面积
    total_intersect = np.sum(intersect_areas)

    # 返回覆盖比例，最大为1.0
    return min(1.0, total_intersect / box_area)



def precision_recall(preds, references, threshold=0.5, workers=8, penalize_double=True):
    """
    计算检测框的precision和recall指标。

    Args:
        preds (list or np.ndarray): 检测出的预测框集合，每个元素为[x1, y1, x2, y2]。
        references (list or np.ndarray): 标注的真实框集合，每个元素为[x1, y1, x2, y2]。
        threshold (float): 判断框是否命中的IOU/覆盖率阈值（0~1之间）。
        workers (int): 并发线程池的最大线程数。
        penalize_double (bool): 是否惩罚多框重叠同一区域。

    Returns:
        dict: {
            "precision": float,  # 精度，预测框命中比例
            "recall": float      # 召回，真实框被命中比例
        }
    """
    # 如果没有标注框，则precision、recall均视作1（边界情况）
    if len(references) == 0:
        return {
            "precision": 1,
            "recall": 1,
        }

    # 如果没有任何预测框，则precision、recall均为0（边界情况）
    if len(preds) == 0:
        return {
            "precision": 0,
            "recall": 0,
        }

    # 根据是否惩罚重叠多框，选择覆盖比例的计算函数
    # 若penalize_double为False，采用快速实现；否则采用标准实现
    coverage_func = calculate_coverage_fast
    if penalize_double:
        coverage_func = calculate_coverage

    # 多线程并发计算
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 计算每个预测框pred被所有标注框references覆盖的比例（即命中真实box的程度）
        precision_func = partial(coverage_func, penalize_double=penalize_double)
        precision_iou = executor.map(precision_func, preds, repeat(references))
        # 计算每个标注框reference被所有预测框preds覆盖的比例（即真实box被命中程度）
        reference_iou = executor.map(coverage_func, references, repeat(preds))

    # 统计精度：预测框被正确命中的数量/总数，i>threshold视为命中
    precision_classes = [1 if i > threshold else 0 for i in precision_iou]
    precision = sum(precision_classes) / len(precision_classes)

    # 统计召回：真实框被正确命中的数量/总数，i>threshold视为命中
    recall_classes = [1 if i > threshold else 0 for i in reference_iou]
    recall = sum(recall_classes) / len(recall_classes)

    # 返回字典格式的precision和recall结果
    return {
        "precision": precision,
        "recall": recall,
    }


def mean_coverage(preds, references):
    coverages = []

    for box1 in references:
        coverage = calculate_coverage(box1, preds)
        coverages.append(coverage)

    for box2 in preds:
        coverage = calculate_coverage(box2, references)
        coverages.append(coverage)

    # Calculate the average coverage over all comparisons
    if len(coverages) == 0:
        return 0
    coverage = sum(coverages) / len(coverages)
    return {"coverage": coverage}


def rank_accuracy(preds, references):
    # Preds and references need to be aligned so each position refers to the same bbox
    pairs = []
    for i, pred in enumerate(preds):
        for j, pred2 in enumerate(preds):
            if i == j:
                continue
            pairs.append((i, j, pred > pred2))

    # Find how many of the prediction rankings are correct
    correct = 0
    for i, ref in enumerate(references):
        for j, ref2 in enumerate(references):
            if (i, j, ref > ref2) in pairs:
                correct += 1

    return correct / len(pairs)
