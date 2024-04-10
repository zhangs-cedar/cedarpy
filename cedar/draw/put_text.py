import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from cedar.draw import color_list


def putText(img, text, position, textColor=tuple(color_list[0]), textSize=30):
    """
    Add text to an image.

    Parameters:
    - img: numpy.ndarray or PIL.Image.Image
        The input image.
    - text: str
        The text to be added.
    - position: tuple (x, y)
        The position (top-left corner) where the text will be added.
    - textColor: tuple (B, G, R), optional
        The color of the text. Default is green (0, 255, 0).
    - textSize: int, optional
        The font size of the text. Default is 30.

    Returns:
    - numpy.ndarray
        The image with the added text.
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
