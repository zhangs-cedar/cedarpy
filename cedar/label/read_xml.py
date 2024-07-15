from xml.etree import cElementTree as et


def xml_to_dict(element):
    """
    将XML Element对象转换为字典。

    Args:
    element (Element): 要转换的XML Element对象。

    Returns:
    dict: 转换后的字典，字典的键为XML元素的标签名，值为对应的文本内容或子元素的字典。

    """
    # 如果当前元素没有子元素，则直接返回文本内容
    if len(element) == 0:
        return element.text

    # 创建字典，用于存储当前元素的属性和子元素
    result = {}

    # 添加属性
    if element.attrib:
        result.update(element.attrib)

    # 遍历子元素
    for child in element:
        # 递归调用xml_to_dict，将子元素转换为字典
        child_dict = xml_to_dict(child)

        # 如果当前子元素已经在字典中，表示有多个相同的子元素，将其转换为列表存储
        if child.tag in result:
            if isinstance(result[child.tag], list):
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = [result[child.tag], child_dict]
        else:
            result[child.tag] = child_dict

    return result


def read_xml(file):
    """
    从给定的XML文件中读取XML内容并返回其根节点。

    Args:
        file (str): XML文件的路径。

    Returns:
        Element: XML文件的根节点。

    """
    tree = et.parse(file)
    root = tree.getroot()
    return root


def read_xml_as_dict(file):
    """
    从给定的XML文件中读取XML内容并返回其根节点对应的字典。
    Args:
        file (str): XML文件的路径。
    Returns:
        dict: 根节点对应的字典。
    """
    root = read_xml(file)
    return xml_to_dict(root)
