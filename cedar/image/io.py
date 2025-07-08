import cv2
import base64
import os.path as osp
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.parse import quote, unquote
from typing import List, Optional


def find_image_path(file_path: str, name: str, extensions: List[str] = ['.png', '.bmp', '.jpg']) -> Optional[str]:
    """在给定文件路径的同一目录下查找图片文件

    根据文件名和扩展名列表查找图片文件的完整路径。

    Args:
        file_path: 参考文件路径
        name: 要查找的图片文件名（不含扩展名）
        extensions: 支持的图片扩展名列表，默认为[".png", ".bmp", ".jpg"]

    Returns:
        Optional[str]: 找到的图片文件完整路径，如果未找到则返回None
    """
    for ext in extensions:
        img_path = osp.join(osp.dirname(file_path), f'{name}{ext}')
        if osp.exists(img_path):
            return img_path
    return None


def is_image(file_path: str) -> bool:
    """判断文件是否为有效的图像文件

    Args:
        file_path: 文件路径

    Returns:
        bool: 如果是有效的图像文件则返回True，否则返回False
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    file_extension = osp.splitext(file_path)[1].lower()
    return file_extension in valid_extensions


def array_to_base64(image_array: np.ndarray) -> str:
    """将NumPy图像数组转换为Base64编码的PNG图像字符串

    Args:
        image_array: 形状为 (height, width, channels) 的NumPy图像数组

    Returns:
        str: 包含Base64编码的PNG图像数据的字符串，格式为 "data:image/png;base64,{img_str}"

    Raises:
        ValueError: 当图像数组为空时
    """
    if image_array is None:
        raise ValueError('图像数组不能为空')

    # 创建PIL图像
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(image_array)

    # 保存图像到字节流
    buffered = BytesIO()
    img.save(buffered, format='PNG')

    # 将字节流转换为Base64编码的字符串
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 返回Base64图像字符串
    return f'data:image/png;base64,{img_str}'


def path_to_url(file_path: str) -> str:
    """将本地路径转换为file URL

    Args:
        file_path: 本地文件路径

    Returns:
        str: 对应的file URL

    Raises:
        ValueError: 当文件路径为空时
    """
    if not file_path:
        raise ValueError('文件路径不能为空')

    # 将本地路径转换为 URL 编码的字符串，但不编码驱动器号
    encoded_path = quote(file_path.replace('\\\\', '/')).replace('%3A', ':').replace('%5C', '/')

    # 构建 file URL
    url = f'file:///{encoded_path}'

    # 修复任何可能的编码问题，如将 %252F 转换为单个 /
    url = url.replace('%252F', '%2F')
    return url


def url_to_path(encoded_url: str) -> str:
    """将编码后的URL转换回本地文件路径

    Args:
        encoded_url: 编码后的file URL

    Returns:
        str: 本地文件路径

    Raises:
        ValueError: 当URL格式不正确时
    """
    if not encoded_url:
        raise ValueError('URL不能为空')

    # 从 URL 中移除 file:/// 前缀
    if encoded_url.startswith('file:///'):
        encoded_path = encoded_url[8:]
    else:
        raise ValueError('提供的 URL 不是有效的 file URL')

    decoded_path = unquote(encoded_path)
    local_path = decoded_path.replace('/', '\\')
    return local_path
