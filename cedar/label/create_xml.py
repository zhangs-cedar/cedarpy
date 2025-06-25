import xml.dom.minidom as minidom
from typing import List, Tuple


def create_xml(
    img_name: str, height: int, width: int, labels: List[str], bboxes: List[List[int]]
) -> minidom.Document:
    """创建XML标注文件

    根据图像信息和标注数据创建符合VOC格式的XML文档对象。

    Args:
        img_name: 图片名称
        height: 图片高度
        width: 图片宽度
        labels: 标签列表，如 ['label1', 'label2', ...]
        bboxes: 边界框列表，格式为 [[xmin, ymin, xmax, ymax], ...]

    Returns:
        minidom.Document: 创建的XML文档对象

    Raises:
        ValueError: 当labels和bboxes长度不匹配时
        IndexError: 当bboxes格式不正确时
    """
    if len(labels) != len(bboxes):
        raise ValueError("labels and bboxes must have the same length")

    xml_doc = minidom.Document()
    root = xml_doc.createElement("annotation")
    xml_doc.appendChild(root)

    # 创建基本信息节点
    _create_basic_info(xml_doc, root, img_name, height, width)

    # 创建标注对象节点
    for i, (label, bbox) in enumerate(zip(labels, bboxes)):
        if len(bbox) != 4:
            raise IndexError(
                f"bbox {i} must have exactly 4 elements: [xmin, ymin, xmax, ymax]"
            )

        xmin, ymin, xmax, ymax = bbox
        _create_object_node(xml_doc, root, label, xmin, ymin, xmax, ymax)

    return xml_doc


def _create_basic_info(
    xml_doc: minidom.Document,
    root: minidom.Element,
    img_name: str,
    height: int,
    width: int,
) -> None:
    """创建XML文档的基本信息节点

    Args:
        xml_doc: XML文档对象
        root: 根节点
        img_name: 图片名称
        height: 图片高度
        width: 图片宽度
    """
    # 创建folder节点
    node_folder = xml_doc.createElement("folder")
    root.appendChild(node_folder)
    node_folder.appendChild(xml_doc.createTextNode("img"))

    # 创建filename节点
    node_filename = xml_doc.createElement("filename")
    root.appendChild(node_filename)
    node_filename.appendChild(xml_doc.createTextNode(img_name))

    # 创建path节点
    path = xml_doc.createElement("path")
    root.appendChild(path)
    path.appendChild(xml_doc.createTextNode(img_name))

    # 创建source节点
    node_source = xml_doc.createElement("source")
    root.appendChild(node_source)
    node_database = xml_doc.createElement("database")
    node_source.appendChild(node_database)
    node_database.appendChild(xml_doc.createTextNode("Unknown"))

    # 创建size节点
    node_size = xml_doc.createElement("size")
    root.appendChild(node_size)

    node_width = xml_doc.createElement("width")
    node_size.appendChild(node_width)
    node_width.appendChild(xml_doc.createTextNode(str(width)))

    node_height = xml_doc.createElement("height")
    node_size.appendChild(node_height)
    node_height.appendChild(xml_doc.createTextNode(str(height)))

    node_depth = xml_doc.createElement("depth")
    node_size.appendChild(node_depth)
    node_depth.appendChild(xml_doc.createTextNode("3"))

    # 创建segmented节点
    node_segmented = xml_doc.createElement("segmented")
    root.appendChild(node_segmented)
    node_segmented.appendChild(xml_doc.createTextNode("0"))


def _create_object_node(
    xml_doc: minidom.Document,
    root: minidom.Element,
    label: str,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> None:
    """创建标注对象节点

    Args:
        xml_doc: XML文档对象
        root: 根节点
        label: 标签名称
        xmin: 边界框左上角x坐标
        ymin: 边界框左上角y坐标
        xmax: 边界框右下角x坐标
        ymax: 边界框右下角y坐标
    """
    node_obj = xml_doc.createElement("object")
    root.appendChild(node_obj)

    # 创建name节点
    node_name = xml_doc.createElement("name")
    node_obj.appendChild(node_name)
    node_name.appendChild(xml_doc.createTextNode(label))

    # 创建pose节点
    node_pose = xml_doc.createElement("pose")
    node_obj.appendChild(node_pose)
    node_pose.appendChild(xml_doc.createTextNode("Unspecified"))

    # 创建truncated节点
    node_truncated = xml_doc.createElement("truncated")
    node_obj.appendChild(node_truncated)
    node_truncated.appendChild(xml_doc.createTextNode("0"))

    # 创建difficult节点
    node_difficult = xml_doc.createElement("difficult")
    node_obj.appendChild(node_difficult)
    node_difficult.appendChild(xml_doc.createTextNode("0"))

    # 创建bndbox节点
    node_box = xml_doc.createElement("bndbox")
    node_obj.appendChild(node_box)

    node_xmin = xml_doc.createElement("xmin")
    node_box.appendChild(node_xmin)
    node_xmin.appendChild(xml_doc.createTextNode(str(xmin)))

    node_ymin = xml_doc.createElement("ymin")
    node_box.appendChild(node_ymin)
    node_ymin.appendChild(xml_doc.createTextNode(str(ymin)))

    node_xmax = xml_doc.createElement("xmax")
    node_box.appendChild(node_xmax)
    node_xmax.appendChild(xml_doc.createTextNode(str(xmax)))

    node_ymax = xml_doc.createElement("ymax")
    node_box.appendChild(node_ymax)
    node_ymax.appendChild(xml_doc.createTextNode(str(ymax)))


def write_xml(xml_doc: minidom.Document, xml_file: str) -> None:
    """将XML对象保存为XML文件

    Args:
        xml_doc: XML文档对象
        xml_file: XML文件保存路径

    Raises:
        IOError: 当文件写入失败时
    """
    with open(xml_file, "w", encoding="utf-8") as f:
        xml_doc.writexml(f, indent="\t", newl="\n", addindent="\t", encoding="utf-8")
