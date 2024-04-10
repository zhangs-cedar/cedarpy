import xml.dom.minidom as minidom


def create_xml(img_name: str, height: int, width: int, labels: list, bboxes: list) -> minidom.Document:
    """创建XML文件
    Args:
        img_name(str): 图片名称
        height(int): 图片高度
        width(int): 图片宽度
        labels(list): 标签列表 ['label1', 'label2', ...]
        bboxes(list): 框列表 [[xmin, ymin, xmax, ymax], ...]
    """

    xml_doc = minidom.Document()
    root = xml_doc.createElement("annotation")
    xml_doc.appendChild(root)
    node_folder = xml_doc.createElement("folder")
    root.appendChild(node_folder)
    node_folder.appendChild(xml_doc.createTextNode("img"))

    node_filename = xml_doc.createElement("filename")
    root.appendChild(node_filename)
    node_filename.appendChild(xml_doc.createTextNode(img_name))

    path = xml_doc.createElement("path")
    root.appendChild(path)
    path.appendChild(xml_doc.createTextNode(img_name))

    node_source = xml_doc.createElement("source")
    root.appendChild(node_source)
    node_database = xml_doc.createElement("database")
    node_source.appendChild(node_database)
    node_database.appendChild(xml_doc.createTextNode("Unknown"))

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
    node_segmented = xml_doc.createElement("segmented")
    root.appendChild(node_segmented)
    node_segmented.appendChild(xml_doc.createTextNode("0"))

    for i in range(len(labels)):
        label = labels[i]
        xmin, ymin, xmax, ymax = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

        node_obj = xml_doc.createElement("object")
        root.appendChild(node_obj)
        node_name = xml_doc.createElement("name")
        node_obj.appendChild(node_name)
        node_name.appendChild(xml_doc.createTextNode(label))
        node_pose = xml_doc.createElement("pose")
        node_obj.appendChild(node_pose)
        node_pose.appendChild(xml_doc.createTextNode("Unspecified"))
        node_truncated = xml_doc.createElement("truncated")
        node_obj.appendChild(node_truncated)
        node_truncated.appendChild(xml_doc.createTextNode("0"))
        node_difficult = xml_doc.createElement("difficult")
        node_obj.appendChild(node_difficult)
        node_difficult.appendChild(xml_doc.createTextNode("0"))

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
    return xml_doc


def write_xml(xml_doc: minidom.Document, xml_file: str):
    """将xml对象保存成xml文件并保存到指定地址
    Args:
        xml_doc(minidom.Document): xml对象
        xml_file(str): xml文件保存路径
    """
    f = open(xml_file, "w", encoding="utf-8")
    xml_doc.writexml(f, indent="\t", newl="\n", addindent="\t", encoding="utf-8")
    f.close()
