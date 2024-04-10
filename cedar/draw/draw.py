import cv2
import numpy as np
from cedar.draw import color_list


def draw_lines(img: np.ndarray, row_lines: object, col_lines: object, color: object, thickness="auto") -> np.ndarray:
    """绘制row、col 线

    Args:
        img (np.ndarray): 图像
        row_lines (object): row 线
        col_lines (object): col 线
        color (object): 颜色
        thickness (str, optional):  Defaults to "auto".

    Returns:
        np.ndarray: 绘制后的图像
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not img.ndim == 3:
        raise ValueError("img must be 3-dim, img.shape:", img.shape)
    if isinstance(color, tuple):
        color = color
    elif isinstance(color, int):
        color = tuple(color_list[color])
    else:
        raise ValueError("color must be tuple or int")
    if thickness == "auto":
        thickness = max(int(img.shape[0] / 1000), 1)

    img_data = img.copy()
    cols = len(col_lines)
    rows = len(row_lines)

    for i in range(0, int(cols)):
        cv2.line(img_data, (col_lines[i], 0), (col_lines[i], img_data.shape[0]), color, thickness)
    for i in range(0, int(rows)):
        cv2.line(img_data, (0, row_lines[i]), (img_data.shape[1], row_lines[i]), color, thickness)
    return img_data
