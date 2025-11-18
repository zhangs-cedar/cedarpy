import cv2
import numpy as np

from cedar.draw import color_list


def draw_lines(img, row_lines, col_lines, color, thickness='auto'):
    """绘制网格线"""
    # 转为3通道
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 处理颜色
    color_bgr = tuple(color_list[color]) if isinstance(color, int) else color
    
    # 处理线宽
    if thickness == 'auto':
        thickness = max(img.shape[0] // 1000, 1)
    
    img_out = img.copy()
    h, w = img_out.shape[:2]
    
    # 绘制垂直线
    for x in col_lines:
        cv2.line(img_out, (x, 0), (x, h), color_bgr, thickness)
    
    # 绘制水平线
    for y in row_lines:
        cv2.line(img_out, (0, y), (w, y), color_bgr, thickness)
    
    return img_out
