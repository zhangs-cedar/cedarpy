import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

from cedar.draw import color_list


def putText(img, text, position, text_color=tuple(color_list[0]), text_size=30):
    """在图像上添加文字"""
    # 转为PIL格式
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img
    
    draw = ImageDraw.Draw(pil_img)
    
    # 加载字体
    try:
        font_path = os.path.join(os.path.dirname(__file__), 'simsun.ttc')
        font = ImageFont.truetype(font_path, text_size, encoding='utf-8')
    except IOError:
        font = ImageFont.load_default()
    
    draw.text(position, text, text_color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
