import cv2
import numpy as np
from typing import List, Tuple, Union
from cedar.draw import color_list


def draw_lines(
    img: np.ndarray,
    row_lines: List[int],
    col_lines: List[int],
    color: Union[Tuple[int, int, int], int],
    thickness: Union[int, str] = "auto",
) -> np.ndarray:
    """绘制行列线

    Args:
        img: 输入图像，shape=(H, W, 3) 或 (H, W)
        row_lines: 行线位置列表
        col_lines: 列线位置列表
        color: 线条颜色，BGR元组或color_list索引
        thickness: 线宽，int或"auto"，默认"auto"

    Returns:
        np.ndarray: 绘制后的图像

    Raises:
        ValueError: 输入参数不合法时抛出
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3:
        raise ValueError(f"img must be 3-dim, got shape: {img.shape}")
    if isinstance(color, tuple):
        color_bgr = color
    elif isinstance(color, int):
        color_bgr = tuple(color_list[color])
    else:
        raise ValueError("color must be tuple or int")
    if thickness == "auto":
        thickness = max(int(img.shape[0] / 1000), 1)
    elif not isinstance(thickness, int):
        raise ValueError("thickness must be int or 'auto'")

    img_data = img.copy()
    for x in col_lines:
        cv2.line(img_data, (x, 0), (x, img_data.shape[0]), color_bgr, thickness)
    for y in row_lines:
        cv2.line(img_data, (0, y), (img_data.shape[1], y), color_bgr, thickness)
    return img_data
