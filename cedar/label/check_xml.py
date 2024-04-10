import numpy as np
import cv2
from xml.etree import cElementTree as et


def check_xml(img_file: object, xml_file: object) -> bool:
    """核对标签文件是否正确,返回True或False

    Args:
        img_file (str/np.ndarray ): 图片文件,兼容np.ndarray和str.
        xml_file (str/et.ElementTree): 标签文件,兼容et.ElementTree和str.

    Returns:
        bool: 标签是否正确
    """
    if isinstance(img_file, str):
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), 1)  # 读取图片、兼容中文路径
    elif isinstance(img_file, np.ndarray):
        img = img_file
    else:
        raise TypeError("img_file must be str or np.ndarray")
    if isinstance(xml_file, str):
        parsexml = et.parse(xml_file)
    elif isinstance(xml_file, et.ElementTree):
        parsexml = xml_file
    else:
        raise TypeError("xml_file must be str or et.ElementTree")

    row_width, col_width = img.shape[:2]
    root = parsexml.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    if row_width != height or col_width != width:
        return False
    if len(root.findall("object")) == 0:
        return False
    for i, obj in enumerate(root.findall("object")):
        bndbox = obj.find("bndbox")
        col_min = int(bndbox.find("xmin").text)
        col_max = int(bndbox.find("xmax").text)
        row_min = int(bndbox.find("ymin").text)
        row_max = int(bndbox.find("ymax").text)
        if col_min < 0 or col_max > width or row_min < 0 or row_max > height:
            return False
        if col_max - col_min < 0 or row_max - row_min < 0:
            return False
    return True
