import cv2
import numpy as np
import cv2
import numpy as np
from cedar.utils.tools import split_filename


def imread(image_path: str, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图片、兼容中文路径、 cv2.IMREAD_COLOR 始终将图像转换为3通道BGR彩色图像。

    Args:
        image_path (str): 图片路径
        flag (int, optional): 图片读取标志。 Defaults to cv2.IMREAD_COLOR,详情见cv2.IMREAD_*。
    Returns:
        np.ndarray: 图片矩阵
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flag)
    return img


def imwrite(image_path: str, img: np.ndarray) -> object:
    """保存图片

    Args:
        image_path (str): 图片路径
        img (np.ndarray): 图片矩阵
    """
    _, suffix = split_filename(image_path)
    cv2.imencode(suffix, img)[1].tofile(image_path)
    return None
