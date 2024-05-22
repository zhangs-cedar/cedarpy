import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from cedar.draw import color_list


def putText(img, text, position, textColor=tuple(color_list[0]), textSize=30):
    """
    在图像上添加文字。
    
    Args:
        img (numpy.ndarray or PIL.Image.Image): 输入图像。
        text (str): 要添加的文字。
        position (tuple (x, y)): 文字添加的位置（左上角坐标）。
        textColor (tuple (B, G, R), optional): 文字的颜色。默认为绿色（0, 255, 0）。
        textSize (int, optional): 文字的字体大小。默认为30。
    
    Returns:
        numpy.ndarray: 添加了文字的图像。
    """

    if isinstance(img, np.ndarray):  # Check if the input image is an OpenCV image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    # Use default font (Arial) if "simsun.ttc" is not available
    try:
        # 获取当前脚本文件的路径
        script_path = os.path.abspath(__file__)
        # 获取当前脚本文件所在的目录
        script_directory = os.path.dirname(script_path)
        font_path = os.path.join(script_directory, "simsun.ttc")
        fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    except IOError:
        fontStyle = ImageFont.load_default()

    draw.text(position, text, textColor, font=fontStyle)  # Draw text on the image
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format
