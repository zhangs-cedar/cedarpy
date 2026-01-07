import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cedar.draw import color_list


def draw_lines(img, row_lines, col_lines, color, thickness='auto'):
    """绘制网格线"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    color_bgr = tuple(color_list[color]) if isinstance(color, int) else color

    if thickness == 'auto':
        thickness = max(img.shape[0] // 1000, 1)

    img_out = img.copy()
    h, w = img_out.shape[:2]

    for x in col_lines:
        cv2.line(img_out, (x, 0), (x, h), color_bgr, thickness)

    for y in row_lines:
        cv2.line(img_out, (0, y), (w, y), color_bgr, thickness)

    return img_out


def putText(img, text, position, text_color=None, text_size=30):
    """在图像上添加文字"""
    if text_color is None:
        text_color = tuple(color_list[0])

    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img

    draw = ImageDraw.Draw(pil_img)

    try:
        font_path = osp.join(osp.dirname(__file__), 'simsun.ttc')
        font = ImageFont.truetype(font_path, text_size, encoding='utf-8')
    except OSError:
        font = ImageFont.load_default()

    draw.text(position, text, text_color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def imshow(img):
    """显示BGR图像"""
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
