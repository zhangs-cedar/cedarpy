import cv2
import numpy as np

from cedar.utils.tools import split_filename


def imread(image_path, flag=cv2.IMREAD_COLOR):
    """读取图片，支持中文路径"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flag)
    if img is None:
        raise FileNotFoundError(f'无法读取: {image_path}')
    return img


def imwrite(image_path, img, plt=False):
    """保存图片，支持中文路径"""
    if img is None:
        raise ValueError('图片不能为空')

    if plt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    suffix = split_filename(image_path)[1]
    cv2.imencode(suffix, img)[1].tofile(image_path)
