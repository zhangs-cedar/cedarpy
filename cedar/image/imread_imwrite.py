import cv2
import numpy as np
from cedar.utils.tools import split_filename


def imread(image_path: str, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图片、兼容中文路径、 cv2.IMREAD_COLOR 始终将图像转换为3通道BGR彩色图像。
        cv2.IMREAD_COLOR：这是默认值，表示图像将以彩色方式读取，即每个像素将由三个通道（BGR）表示。
        cv2.IMREAD_GRAYSCALE：这个标志表示图像将以灰度方式读取，即每个像素将只有一个通道，表示其亮度信息。
        cv2.IMREAD_UNCHANGED：这个标志表示图像将包括其 alpha 通道（如果存在）一起读取。这对于处理带有透明度的图像（如PNG格式）是必要的。
    Args:
        image_path (str): 图片路径
        flag (int, optional): 图片读取标志。 Defaults to cv2.IMREAD_COLOR,详情见cv2.IMREAD_*。
    Returns:
        np.ndarray: 图片矩阵
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flag)
    return img


def imwrite(image_path: str, img: np.ndarray,plt=False) -> object:
    """保存图片,如果plt则先转通道再保存

    Args:
        image_path (str): 图片路径
        img (np.ndarray): 图片矩阵
    """
    if plt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, suffix = split_filename(image_path)
    cv2.imencode(suffix, img)[1].tofile(image_path)
    return None
