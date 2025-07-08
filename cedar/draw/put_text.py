import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Union
from cedar.draw import color_list


def putText(
    img: Union[np.ndarray, Image.Image],
    text: str,
    position: Tuple[int, int],
    text_color: Tuple[int, int, int] = tuple(color_list[0]),
    text_size: int = 30,
) -> np.ndarray:
    """在图像上添加文字

    Args:
        img: 输入图像，np.ndarray 或 PIL.Image.Image
        text: 要添加的文字
        position: 文字添加的位置（左上角坐标）
        text_color: 文字颜色，BGR元组，默认绿色
        text_size: 字体大小，默认30

    Returns:
        np.ndarray: 添加了文字的BGR格式图像
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif not isinstance(img, Image.Image):
        raise ValueError('img must be np.ndarray or PIL.Image.Image')

    draw = ImageDraw.Draw(img)
    # Use default font (Arial) if "simsun.ttc" is not available
    try:
        # 获取当前脚本文件的路径
        script_path = os.path.abspath(__file__)
        # 获取当前脚本文件所在的目录
        script_directory = os.path.dirname(script_path)
        font_path = os.path.join(script_directory, 'simsun.ttc')
        font_style = ImageFont.truetype(font_path, text_size, encoding='utf-8')
    except IOError:
        font_style = ImageFont.load_default()

    draw.text(position, text, text_color, font=font_style)  # Draw text on the image
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format
