import cv2
import numpy as np
from typing import Optional
from cedar.utils.tools import split_filename


def imread(image_path: str, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图片，兼容中文路径

    支持多种读取模式：
    - cv2.IMREAD_COLOR: 默认值，将图像转换为3通道BGR彩色图像
    - cv2.IMREAD_GRAYSCALE: 以灰度方式读取，每个像素只有一个通道
    - cv2.IMREAD_UNCHANGED: 包括alpha通道一起读取，适用于PNG等透明图像

    Args:
        image_path: 图片路径
        flag: 图片读取标志，默认为cv2.IMREAD_COLOR

    Returns:
        np.ndarray: 图片矩阵

    Raises:
        FileNotFoundError: 当图片文件不存在时
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flag)
    if img is None:
        raise FileNotFoundError(f'无法读取图片文件: {image_path}')
    return img


def imwrite(image_path: str, img: np.ndarray, plt: bool = False) -> None:
    """保存图片

    Args:
        image_path: 图片保存路径
        img: 图片矩阵
        plt: 是否转换为RGB格式保存，默认为False

    Raises:
        ValueError: 当图片矩阵为空时
    """
    if img is None:
        raise ValueError('图片矩阵不能为空')

    if plt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _, suffix = split_filename(image_path)
    cv2.imencode(suffix, img)[1].tofile(image_path)
