import numpy as np
import cv2
from xml.etree import cElementTree as et
from typing import Union, Tuple


def check_xml(
    img_file: Union[str, np.ndarray], xml_file: Union[str, et.ElementTree]
) -> bool:
    """核对标签文件是否正确

    检查XML标签文件中的标注框是否与图像尺寸匹配，以及标注框是否在有效范围内。

    Args:
        img_file: 图片文件，支持文件路径字符串或numpy数组
        xml_file: 标签文件，支持文件路径字符串或ElementTree对象

    Returns:
        bool: 标签是否正确，True表示正确，False表示有错误

    Raises:
        TypeError: 当输入参数类型不正确时
        FileNotFoundError: 当文件不存在时
    """
    # 处理图像文件
    if isinstance(img_file, str):
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), 1)
    elif isinstance(img_file, np.ndarray):
        img = img_file
    else:
        raise TypeError("img_file must be str or np.ndarray")

    # 处理XML文件
    if isinstance(xml_file, str):
        parsexml = et.parse(xml_file)
    elif isinstance(xml_file, et.ElementTree):
        parsexml = xml_file
    else:
        raise TypeError("xml_file must be str or et.ElementTree")

    # 获取图像尺寸
    row_width, col_width = img.shape[:2]

    # 获取XML根节点
    root = parsexml.getroot()

    # 获取XML中记录的图像尺寸
    size_element = root.find("size")
    if size_element is None:
        return False

    width_element = size_element.find("width")
    height_element = size_element.find("height")

    if width_element is None or height_element is None:
        return False

    width = int(width_element.text)
    height = int(height_element.text)

    # 检查图像尺寸是否匹配
    if row_width != height or col_width != width:
        return False

    # 检查是否有标注对象
    objects = root.findall("object")
    if len(objects) == 0:
        return False

    # 检查每个标注框
    for obj in objects:
        bndbox = obj.find("bndbox")
        if bndbox is None:
            return False

        # 获取边界框坐标
        xmin_element = bndbox.find("xmin")
        xmax_element = bndbox.find("xmax")
        ymin_element = bndbox.find("ymin")
        ymax_element = bndbox.find("ymax")

        if any(
            elem is None
            for elem in [xmin_element, xmax_element, ymin_element, ymax_element]
        ):
            return False

        col_min = int(xmin_element.text)
        col_max = int(xmax_element.text)
        row_min = int(ymin_element.text)
        row_max = int(ymax_element.text)

        # 检查边界框是否在图像范围内
        if col_min < 0 or col_max > width or row_min < 0 or row_max > height:
            return False

        # 检查边界框是否有效（宽高大于0）
        if col_max - col_min < 0 or row_max - row_min < 0:
            return False

    return True
