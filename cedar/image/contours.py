"""
轮廓处理模块

提供图像轮廓检测、向量计算、角度计算等功能。
mask: 0/255 mask of the image, 0 for the background, 255 for the foreground
"""

import cv2
import numpy as np
from typing import List, Tuple, Union


def get_contours(mask: np.ndarray) -> List[np.ndarray]:
    """获取掩码图像的轮廓

    Args:
        mask: 二值掩码图像，0表示背景，255表示前景

    Returns:
        List[np.ndarray]: 轮廓点列表，每个轮廓是一个numpy数组
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_vertios(point1: List[float], point2: List[float]) -> List[float]:
    """根据两个点计算向量

    向量的坐标系为直角坐标系，x轴为水平方向，y轴为垂直方向。
    注：y需要取负值，因为y轴向上为正。

    Args:
        point1: 第一个点的坐标，格式为 [x, y]
        point2: 第二个点的坐标，格式为 [x, y]

    Returns:
        List[float]: 两个点的向量，格式为 [x, y]
    """
    if point1[0] > point2[0]:
        v = [point1[0] - point2[0], point2[1] - point1[1]]
    else:
        v = [point2[0] - point1[0], point1[1] - point2[1]]
    return v


def get_longside_rect_ps(rect_ps: np.ndarray) -> List[float]:
    """获取矩形长边的向量

    Args:
        rect_ps: 矩形四个点的坐标数组

    Returns:
        List[float]: 长边向量，格式为 [x, y]
    """
    p0 = rect_ps[0]
    p1 = rect_ps[1]
    p2 = rect_ps[-1]

    dis1 = np.linalg.norm(p0 - p1)
    dis2 = np.linalg.norm(p0 - p2)

    if dis1 > dis2:
        return get_vertios(p0.tolist(), p1.tolist())
    else:
        return get_vertios(p0.tolist(), p2.tolist())


def calcu_angle_between_verctors(v1: List[float], v2: List[float]) -> float:
    """计算两个向量之间的夹角

    Args:
        v1: 第一个向量，格式为 [x, y]
        v2: 第二个向量，格式为 [x, y]

    Returns:
        float: 两个向量之间的夹角，单位为度，范围为0-180度
    """
    # 计算向量的点积
    dot_product = np.dot(v1, v2)
    # 计算向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 计算夹角的弧度值
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))
    # 将弧度值转换为角度值
    angle_deg = np.degrees(angle)
    return angle_deg


def calcu_angle(v1: List[float]) -> float:
    """计算向量与x轴正方向的夹角

    Args:
        v1: 向量，格式为 [x, y]

    Returns:
        float: 向量与x轴正方向的夹角，单位为度
    """
    v2 = [1, 0]
    angle_deg = calcu_angle_between_verctors(v1, v2)
    return angle_deg


def get_minAreaRect(
    cnt: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """获取轮廓的最小外接矩形

    Args:
        cnt: 轮廓点数组

    Returns:
        Tuple: 最小外接矩形信息，包含中心点、宽高和角度
    """
    rect = cv2.minAreaRect(cnt)
    return rect
