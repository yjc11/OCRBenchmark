# -*- coding: utf-8 -*-
# File: common.py

import copy
import math
import os
import random
from functools import cmp_to_key

import cv2
import numpy as np
import pycocotools.mask as cocomask
from shapely.geometry import Polygon


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def cmp(a, b, c):
    """
    输入参数：
    ----------------------
    多个相交点的逆时针排序
    a,b是要进行比较的两个点
    c是所有点的x、y的均值

    输出参数：
    ---------------------
    -1: a < b
    0 : a = b
    1 : a > b
    """
    if a.x >= 0 and b.x < 0:
        return -1
    if a.x == 0 and b.x == 0:
        # return a.y > b.y
        if a.y > b.y:
            return -1
        elif a.y < b.y:
            return 1
        return 0
    det = (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y)
    if det < 0:
        return 1
    if det > 0:
        return -1
    d1 = (a.x - c.x) * (a.x - c.x) + (a.y - c.y) * (a.y - c.y)
    d2 = (b.x - c.x) * (b.x - c.x) + (b.y - c.y) * (b.y - c.y)
    # return d1 > d2
    if d1 > d2:
        return -1
    elif d1 < d2:
        return 1
    return 0


def box_to_point8(boxes):
    """
    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)  # nx2
    maxxy = p.max(axis=1)  # nx2
    return np.concatenate((minxy, maxxy), axis=1)


def polygons_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    import pycocotools.mask as cocomask

    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def filter_boxes_inside_shape(boxes, shape):
    """
    Args:
        boxes: (nx4), float
        shape: (h, w)

    Returns:
        indices: (k, )
        selection: (kx4)
    """
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= w) & (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]


# Much faster than utils/np_box_ops
def np_iou(A, B):
    def to_xywh(box):
        box = box.copy()
        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]
        return box

    ret = cocomask.iou(to_xywh(A), to_xywh(B), np.zeros((len(B),), dtype=np.bool))
    # can accelerate even more, if using float32
    return ret.astype('float32')


def rotate_image_only(im, angle):
    """
    rotate image in range[-10,10]
    :param polys:
    :param tags:
    :return:
    """

    def rotate(src, angle, scale=1.0):  # 1
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rotated_image = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        return rotated_image

    old_h, old_w, _ = im.shape
    old_center = (old_w / 2, old_h / 2)

    image = rotate(im, angle)
    new_h, new_w, _ = image.shape
    new_center = (new_w / 2, new_h / 2)

    return image, old_center, new_center


def rotate_polys_only(old_center, new_center, polys, angle):
    """
    rotate image in range[-10,10]
    :param polys:
    :param tags:
    :return:
    """
    polys_copy = copy.deepcopy(polys)
    angle = angle * np.pi * 1.0 / 180
    new_polys = []
    for poly in polys_copy:
        # print('poly:', poly)
        poly[:, 0] = poly[:, 0] - new_center[0]
        poly[:, 1] = new_center[1] - poly[:, 1]
        x1 = poly[0, 0] * math.cos(angle) - poly[0, 1] * math.sin(angle) + old_center[0]
        y1 = old_center[1] - (poly[0, 0] * math.sin(angle) + poly[0, 1] * math.cos(angle))
        x2 = poly[1, 0] * math.cos(angle) - poly[1, 1] * math.sin(angle) + old_center[0]
        y2 = old_center[1] - (poly[1, 0] * math.sin(angle) + poly[1, 1] * math.cos(angle))
        x3 = poly[2, 0] * math.cos(angle) - poly[2, 1] * math.sin(angle) + old_center[0]
        y3 = old_center[1] - (poly[2, 0] * math.sin(angle) + poly[2, 1] * math.cos(angle))
        x4 = poly[3, 0] * math.cos(angle) - poly[3, 1] * math.sin(angle) + old_center[0]
        y4 = old_center[1] - (poly[3, 0] * math.sin(angle) + poly[3, 1] * math.cos(angle))
        new_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    return np.array(new_polys, dtype=np.float32)


def compute_angle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        return 0
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0])
        )
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
            return angle
        if angle / np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            return -(np.pi / 2 - angle)
        else:
            # 这个点为p3 - this point is p3
            return angle


def rotate_image90(im, polys, angle):
    """
    random rotate image 90, -90
    :param polys:
    :param tags:
    :return:
    """
    (h, w) = im.shape[:2]
    center = (w / 2, h / 2)  # x在前，y在后
    new_center = (h / 2, w / 2)  # 旋转之后新的坐标中心，x在前，y在后，高变宽，宽变高
    if angle == 90:
        image = np.rot90(im, k=1)
    else:
        image = np.rot90(im, k=-1)

    angle = angle * np.pi * 1.0 / 180
    new_polys = []
    for poly in polys:
        # print('poly:', poly)
        poly[:, 0] = poly[:, 0] - center[0]
        poly[:, 1] = center[1] - poly[:, 1]
        x1 = poly[0, 0] * math.cos(angle) - poly[0, 1] * math.sin(angle) + new_center[0]
        y1 = new_center[1] - (poly[0, 0] * math.sin(angle) + poly[0, 1] * math.cos(angle))
        x2 = poly[1, 0] * math.cos(angle) - poly[1, 1] * math.sin(angle) + new_center[0]
        y2 = new_center[1] - (poly[1, 0] * math.sin(angle) + poly[1, 1] * math.cos(angle))
        x3 = poly[2, 0] * math.cos(angle) - poly[2, 1] * math.sin(angle) + new_center[0]
        y3 = new_center[1] - (poly[2, 0] * math.sin(angle) + poly[2, 1] * math.cos(angle))
        x4 = poly[3, 0] * math.cos(angle) - poly[3, 1] * math.sin(angle) + new_center[0]
        y4 = new_center[1] - (poly[3, 0] * math.sin(angle) + poly[3, 1] * math.cos(angle))
        new_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    return image, np.array(new_polys, dtype=np.float32)


def rotate_image(im, polys, angle):
    """
    rotate image in range[-20,20]
    :param polys:
    :param tags:
    :return:
    """

    def rotate(src, angle, scale=1.0):  # 1
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rotated_image = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        return rotated_image

    old_h, old_w, _ = im.shape
    old_center = (old_w / 2, old_h / 2)

    image = rotate(im, angle)
    new_h, new_w, _ = image.shape
    new_center = (new_w / 2, new_h / 2)

    angle = angle * np.pi * 1.0 / 180
    new_polys = []
    for poly in polys:
        # print('poly:', poly)
        poly[:, 0] = poly[:, 0] - old_center[0]
        poly[:, 1] = old_center[1] - poly[:, 1]
        x1 = poly[0, 0] * math.cos(angle) - poly[0, 1] * math.sin(angle) + new_center[0]
        y1 = new_center[1] - (poly[0, 0] * math.sin(angle) + poly[0, 1] * math.cos(angle))
        x2 = poly[1, 0] * math.cos(angle) - poly[1, 1] * math.sin(angle) + new_center[0]
        y2 = new_center[1] - (poly[1, 0] * math.sin(angle) + poly[1, 1] * math.cos(angle))
        x3 = poly[2, 0] * math.cos(angle) - poly[2, 1] * math.sin(angle) + new_center[0]
        y3 = new_center[1] - (poly[2, 0] * math.sin(angle) + poly[2, 1] * math.cos(angle))
        x4 = poly[3, 0] * math.cos(angle) - poly[3, 1] * math.sin(angle) + new_center[0]
        y4 = new_center[1] - (poly[3, 0] * math.sin(angle) + poly[3, 1] * math.cos(angle))
        new_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    return image, np.array(new_polys, dtype=np.float32)


def data_augment_rotate_image(im, segmentation, klass, is_crowd, rotate_90, rotate_small_angle, small_angle):
    """
    对图片进行随机（-30，30）度旋转，并变换box坐标
    Args:
        im, segmentation, klass, is_crowd
    Return:
        im, boxes, segmentation, klass, is_crowd
    """
    rotate_boxes = []
    rotate_segmentation = []
    rotate_klass = []
    rotate_is_crowd = []
    text_polys = [segmentation[k][0] for k in range(len(segmentation))]  # n*4*2
    # 以0.5的概率随机旋转90，-90，180
    if rotate_90:
        if np.random.rand() < 1.0 / 2:
            angle = np.random.choice(np.array([-90, 90, 180]))
            if angle == 180:
                im, text_polys = rotate_image90(im, text_polys, 90)
                im, text_polys = rotate_image90(im, text_polys, 90)
            else:
                im, text_polys = rotate_image90(im, text_polys, angle)

    # 增加-small_angle到small_angle的随机旋转
    if rotate_small_angle:
        angle = random.uniform(-1 * small_angle, small_angle)
        im, text_polys = rotate_image(im, np.array(text_polys), angle)
    h, w, _ = im.shape
    text_polys[:, :, 0] = np.clip(text_polys[:, :, 0], 0, w - 1)
    text_polys[:, :, 1] = np.clip(text_polys[:, :, 1], 0, h - 1)
    for index, text_poly in enumerate(text_polys):
        x1 = min(text_poly[:, 0])
        x2 = max(text_poly[:, 0])
        y1 = min(text_poly[:, 1])
        y2 = max(text_poly[:, 1])
        if (x2 - x1 >= 1) and (y2 - y1 >= 1):
            rotate_boxes.append([x1, y1, x2, y2])
            rotate_is_crowd.append(is_crowd[index])
            rotate_klass.append(klass[index])

            seg = [np.array(text_poly).astype('float32')]
            rotate_segmentation.append(seg)

    boxes = np.array(rotate_boxes, dtype='float32')
    segmentation = rotate_segmentation
    klass = np.array(rotate_klass)
    is_crowd = np.array(rotate_is_crowd)

    return im, boxes, segmentation, klass, is_crowd


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1.0, 0.0, -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1.0, b]


def line_cross_point(line1_points, line2_points):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    line1_point1 = line1_points[0]
    line1_point2 = line1_points[1]
    line2_point1 = line2_points[0]
    line2_point2 = line2_points[1]
    # 初步判断line1和line2是否有交集
    line1_x_range = np.array([min(line1_point1[0], line1_point2[0]), max(line1_point1[0], line1_point2[0])])
    line1_y_range = np.array([min(line1_point1[1], line1_point2[1]), max(line1_point1[1], line1_point2[1])])
    line2_x_range = np.array([min(line2_point1[0], line2_point2[0]), max(line2_point1[0], line2_point2[0])])
    line2_y_range = np.array([min(line2_point1[1], line2_point2[1]), max(line2_point1[1], line2_point2[1])])
    if (
        (line2_x_range[1] < line1_x_range[0])
        or (line2_x_range[0] > line1_x_range[1])
        or (line2_y_range[1] < line1_y_range[0])
        or (line2_y_range[0] > line1_y_range[1])
    ):
        return []
    line1 = fit_line([line1_point1[0], line1_point2[0]], [line1_point1[1], line1_point2[1]])
    line2 = fit_line([line2_point1[0], line2_point2[0]], [line2_point1[1], line2_point2[1]])

    if line1[0] != 0 and line1[0] == line2[0]:
        return []
    if line1[0] == 0 and line2[0] == 0:
        return []
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1

    if (
        (x >= (min(line1_point1[0], line1_point2[0]) - 0.1) and x <= (max(line1_point1[0], line1_point2[0])) + 0.1)
        and (x >= (min(line2_point1[0], line2_point2[0]) - 0.1) and x <= (max(line2_point1[0], line2_point2[0])) + 0.1)
        and (y >= (min(line1_point1[1], line1_point2[1]) - 0.1) and y <= (max(line1_point1[1], line1_point2[1])) + 0.1)
        and (y >= (min(line2_point1[1], line2_point2[1]) - 0.1) and y <= (max(line2_point1[1], line2_point2[1])) + 0.1)
    ):
        return np.array([x, y], dtype=np.float32)
    else:
        return []


def vertex_in_cnt(cnt1, cnt2):
    vertex_coordinates = []
    for points in cnt1:
        flag = int(cv2.pointPolygonTest(np.array(cnt2), (points[0], points[1]), False))
        if flag == 1:
            vertex_coordinates.append(points)
    for points in cnt2:
        flag = int(cv2.pointPolygonTest(np.array(cnt1), (points[0], points[1]), False))
        if flag == 1:
            vertex_coordinates.append(points)
    return vertex_coordinates


def sort_points(vertex_coordinates):
    x = 0
    y = 0
    p = []
    len_p = len(vertex_coordinates)
    for i in range(len_p):
        p.append(Point(vertex_coordinates[i][0], vertex_coordinates[i][1]))
        x += vertex_coordinates[i][0]
        y += vertex_coordinates[i][1]

    c = Point(x / len_p, y / len_p)

    pp = sorted(p, key=cmp_to_key(lambda x, y: cmp(x, y, c)))
    r = np.full((len_p, 2), 0.0, dtype='float32')
    for i in range(len(pp)):
        # print (pp[i].x, pp[i].y)
        r[i][0] = pp[i].x
        r[i][1] = pp[i].y
    return r


def data_augment_crop_area(im, input_size, segmentation, klass, is_crowd):
    '''
    随机crop input_size*input_size区域，并变换box坐标信息，旋转box可能出现被截断情况
    Args:
        im, segmentation, klass, is_crowd
    Return:
        im, boxes, segmentation, klass, is_crowd
    '''
    text_polys = [segmentation[k][0] for k in range(len(segmentation))]
    resize_h, resize_w, _ = im.shape
    crop_h = min(resize_h, input_size)
    crop_w = min(resize_w, input_size)
    h_axis = resize_h - crop_h
    w_axis = resize_w - crop_w
    ymin = random.randint(0, h_axis)
    xmin = random.randint(0, w_axis)
    im = im[ymin : ymin + crop_h, xmin : xmin + crop_w, :]
    crop_cnt = np.array(
        [[xmin, ymin], [xmin + crop_w - 1, ymin], [xmin + crop_w - 1, ymin + crop_h - 1], [xmin, ymin + crop_h - 1]],
        dtype=np.float32,
    )
    crop_lines = [
        [crop_cnt[0], crop_cnt[1]],
        [crop_cnt[1], crop_cnt[2]],
        [crop_cnt[2], crop_cnt[3]],
        [crop_cnt[3], crop_cnt[0]],
    ]

    crop_boxes = []
    crop_segmentation = []
    crop_klass = []
    crop_is_crowd = []
    crop_cos = []
    crop_sin = []
    for index, text_poly in enumerate(text_polys):
        # compute angle, 默认起始点是从文字的左上角给出的
        angle_vector = text_poly[1] - text_poly[0]
        cos_angle = (angle_vector[0] / np.linalg.norm(angle_vector) + 1) / 2
        sin_angle = (angle_vector[1] / np.linalg.norm(angle_vector) + 1) / 2

        text_cnt = np.array(text_poly, dtype=np.float32)
        vertex_coordinates = vertex_in_cnt(text_cnt, crop_cnt)
        text_lines = [
            [text_poly[0], text_poly[1]],
            [text_poly[1], text_poly[2]],
            [text_poly[2], text_poly[3]],
            [text_poly[3], text_poly[0]],
        ]
        for text_line in text_lines:
            for crop_line in crop_lines:
                cross_point = line_cross_point(text_line, crop_line)
                if len(cross_point):
                    vertex_coordinates.append(cross_point)

        if len(vertex_coordinates):
            # 去重
            vertex_coordinates = np.unique(vertex_coordinates, axis=0)
            points = sort_points(vertex_coordinates)

            if len(points) >= 3:
                points[:, 0] -= xmin
                points[:, 1] -= ymin
                points[:, 0] = np.clip(points[:, 0], 0, crop_w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, crop_h - 1)

                x1 = min(points[:, 0])
                x2 = max(points[:, 0])
                y1 = min(points[:, 1])
                y2 = max(points[:, 1])
                if (x2 - x1 >= 1) and (y2 - y1 >= 1):
                    crop_boxes.append([x1, y1, x2, y2])
                    crop_is_crowd.append(is_crowd[index])
                    crop_klass.append(klass[index])
                    seg = [np.array(points).astype('float32')]
                    crop_segmentation.append(seg)
                    crop_cos.append(cos_angle)
                    crop_sin.append(sin_angle)

    boxes = np.array(crop_boxes, dtype='float32')
    segmentation = crop_segmentation
    klass = np.array(crop_klass)
    is_crowd = np.array(crop_is_crowd)
    boxes_cos = np.array(crop_cos)
    boxes_sin = np.array(crop_sin)

    return im, boxes, segmentation, klass, is_crowd, boxes_cos, boxes_sin


def compute_cos_sin(segmentation):
    """
    compute cos and sin of bouding box
    Args:
        segmentation
    Return:
        boxes_cos, boxes_sin
    """
    text_polys = [segmentation[k][0] for k in range(len(segmentation))]
    boxes_cos = []
    boxes_sin = []
    for index, text_poly in enumerate(text_polys):
        # compute angle, 默认起始点是从文字的左上角给出的
        angle_vector = text_poly[1] - text_poly[0]
        cos_angle = (angle_vector[0] / np.linalg.norm(angle_vector) + 1) / 2
        sin_angle = (angle_vector[1] / np.linalg.norm(angle_vector) + 1) / 2
        boxes_cos.append(cos_angle)
        boxes_sin.append(sin_angle)

    boxes_cos = np.array(boxes_cos)
    boxes_sin = np.array(boxes_sin)
    return boxes_cos, boxes_sin


def find_contours(mask, method=None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype=np.uint8)
    mask = mask.copy()
    # mode = cv2.RETR_CCOMP
    mode = cv2.RETR_EXTERNAL
    try:
        contours, _ = cv2.findContours(mask, mode=mode, method=method)
    except:
        _, contours, _ = cv2.findContours(mask, mode=mode, method=method)
    return contours


def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h


def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]

    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    # points = np.int0(points)
    # for i_xy, (x, y) in enumerate(points):
    #     x = get_valid_x(x)
    #     y = get_valid_y(y)
    #     points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points


def mask_to_bboxes(masks, scores, boxes_cos, boxes_sin, image_shape=None, labels=[]):
    # Minimal shorter side length and area are used for post- filtering and set to 10 and 300 respectively
    min_area = 0
    min_height = 0

    valid_scores = []
    valid_boxes_cos = []
    valid_boxes_sin = []
    valid_labels = []

    bboxes = []
    max_bbox_idx = len(masks)

    for bbox_idx in range(0, max_bbox_idx):
        bbox_mask = masks[bbox_idx, :, :]
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        # 只回归最大面积的mask的box
        max_area = 0
        max_index = 0
        for index, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_index = index
        cnt = cnts[max_index]
        rect, rect_area = min_area_rect(cnt)
        w, h = rect[2:-1]
        if min(w, h) <= min_height:
            continue

        if rect_area <= min_area:
            continue

        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
        valid_scores.append(scores[bbox_idx])
        valid_boxes_cos.append(boxes_cos[bbox_idx])
        valid_boxes_sin.append(boxes_sin[bbox_idx])
        if len(labels) > 0:
            valid_labels.append(labels[bbox_idx])

    if len(labels) > 0:
        return bboxes, valid_scores, valid_boxes_cos, valid_boxes_sin, valid_labels
    else:
        return bboxes, valid_scores, valid_boxes_cos, valid_boxes_sin


def start_point_boxes(boxes, boxes_cos, boxes_sin):
    """
    确定boxes起始点，第一个点总是左上点
    """
    for box_index in range(len(boxes)):
        cos_value = boxes_cos[box_index]
        sin_value = boxes_sin[box_index]
        cos_value_norm = 2 * cos_value - 1
        sin_value_norm = 2 * sin_value - 1
        cos_value = cos_value_norm / math.sqrt(math.pow(cos_value_norm, 2) + math.pow(sin_value_norm, 2))
        sin_value = sin_value_norm / math.sqrt(math.pow(cos_value_norm, 2) + math.pow(sin_value_norm, 2))
        # print('cos_value:', cos_value, 'sin_value:', sin_value)

        cos_angle = math.acos(cos_value) * 180 / np.pi
        sin_angle = math.asin(sin_value) * 180 / np.pi
        if cos_angle <= 90 and sin_angle <= 0:
            angle = 360 + sin_angle
        elif cos_angle <= 90 and sin_angle > 0:
            angle = sin_angle
        elif cos_angle > 90 and sin_angle > 0:
            angle = cos_angle
        elif cos_angle > 90 and sin_angle <= 0:
            angle = 360 - cos_angle
        # print('angle:', angle)

        box = boxes[box_index]
        box = box[:8].reshape((4, 2))
        box_angle_vector = box[1] - box[0]
        box_cos_value = box_angle_vector[0] / np.linalg.norm(box_angle_vector)
        box_sin_value = box_angle_vector[1] / np.linalg.norm(box_angle_vector)
        box_cos_angle = math.acos(box_cos_value) * 180 / np.pi
        box_sin_angle = math.asin(box_sin_value) * 180 / np.pi
        if box_cos_angle <= 90 and box_sin_angle <= 0:
            box_angle = 360 + box_sin_angle
        elif box_cos_angle <= 90 and box_sin_angle > 0:
            box_angle = box_sin_angle
        elif box_cos_angle > 90 and box_sin_angle > 0:
            box_angle = box_cos_angle
        elif box_cos_angle > 90 and box_sin_angle <= 0:
            box_angle = 360 - box_cos_angle
        box_angle = np.array([box_angle, (box_angle + 90) % 360, (box_angle + 180) % 360, (box_angle + 270) % 360])

        delta_angle = np.append(np.abs(box_angle - angle), 360 - np.abs(box_angle - angle))
        start_point_index = np.argmin(delta_angle) % 4
        box = box[
            [start_point_index, (start_point_index + 1) % 4, (start_point_index + 2) % 4, (start_point_index + 3) % 4]
        ]
        boxes[box_index] = box.reshape((-1))
    return boxes


def order_points(pts):
    """
    对坐标点进行排序
    """
    pts = np.array(pts)
    xSorted = pts[np.argsort(pts[:, 0]), :]
    if xSorted[1][0] == xSorted[2][0] and xSorted[1][1] >= xSorted[2][1]:
        xSorted = xSorted[[0, 2, 1, 3], :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    return np.array([tl, tr, br, bl])


def intersection(g, p):
    """
    compute iou
    """
    g = Polygon(g)
    p = Polygon(p)
    if not g.is_valid or not p.is_valid:
        return 0, 0
    inter = Polygon(g).intersection(Polygon(p)).area
    p_in_g = inter / p.area
    union = g.area + p.area - inter
    if union == 0:
        return 0, 0
    else:
        return inter / union, p_in_g


def perspective_transform(img, pts):
    def dist_euclid(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    W = int(dist_euclid(pts[0], pts[1])) + 1
    H = int(dist_euclid(pts[1], pts[2])) + 1

    pts = np.array(pts, 'float32')
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], 'float32')
    M0 = cv2.getPerspectiveTransform(pts, dst)
    image = cv2.warpPerspective(img, M0, (W, H))
    return image


def write_lines(p, lines, append_break=False):
    with open(p, 'w') as f:
        for line in lines:
            if append_break:
                f.write(line + '\n')
            else:
                f.write(line)


def write_result_as_txt(image_name, bboxes, path, labels=[]):
    filename = os.path.join(path, '%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        # values = [float(v) for v in bbox]
        if len(labels) > 0:
            line = "%d, %d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values + [labels[b_idx]])
        else:
            line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        # line = "%f, %f, %f, %f, %f, %f, %f, %f, %f\n"%tuple(values)
        lines.append(line)
    write_lines(filename, lines)
    print('result has been written to:', filename)
