import cv2
import numpy as np
from typing import Tuple, Union


def roate_image(img: np.ndarray, angle: float, border_value: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """对图像进行旋转操作

    目前仅支持正方形图像的旋转操作，且旋转角度为正数表示逆时针旋转，
    负数表示顺时针旋转。

    Args:
        img: 待旋转的图像，应为numpy数组类型
        angle: 旋转角度，单位为度，正数表示逆时针旋转，负数表示顺时针旋转
        border_value: 旋转后图像边界外的填充颜色，默认为黑色(0, 0, 0)

    Returns:
        np.ndarray: 旋转后的图像，返回numpy数组类型

    Raises:
        ValueError: 当图像为空时
    """
    if img is None:
        raise ValueError('图像不能为空')

    h, w = img.shape[:2]
    c_x, c_y = w // 2, h // 2

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算新图像的尺寸
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵的平移部分
    M[0, 2] += (n_w / 2) - c_x
    M[1, 2] += (n_h / 2) - c_y

    # 执行旋转变换
    rotated_img = cv2.warpAffine(img, M, (n_w, n_h), flags=cv2.INTER_NEAREST, borderValue=border_value)

    # 裁剪到原始尺寸
    p_w = (n_w - w) // 2
    p_h = (n_h - h) // 2

    return rotated_img[p_h : p_h + h, p_w : p_w + w]
