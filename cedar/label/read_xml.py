from xml.etree import cElementTree as et
from typing import Dict, Any, Union


def xml_to_dict(element: et.Element) -> Union[Dict[str, Any], str]:
    """将XML Element对象转换为字典

    递归地将XML元素及其子元素转换为Python字典结构。
    如果元素没有子元素，则返回其文本内容。

    Args:
        element: 要转换的XML Element对象

    Returns:
        Union[Dict[str, Any], str]: 转换后的字典或字符串
    """
    # 如果当前元素没有子元素，则直接返回文本内容
    if len(element) == 0:
        return element.text

    # 创建字典，用于存储当前元素的属性和子元素
    result: Dict[str, Any] = {}

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


def read_xml(file_path: str) -> et.Element:
    """从XML文件中读取内容并返回根节点

    Args:
        file_path: XML文件的路径

    Returns:
        et.Element: XML文件的根节点

    Raises:
        FileNotFoundError: 当文件不存在时
        et.ParseError: 当XML格式错误时
    """
    tree = et.parse(file_path)
    root = tree.getroot()
    return root


def read_xml_as_dict(file_path: str) -> Dict[str, Any]:
    """从XML文件中读取内容并转换为字典

    Args:
        file_path: XML文件的路径

    Returns:
        Dict[str, Any]: 根节点对应的字典

    Raises:
        FileNotFoundError: 当文件不存在时
        et.ParseError: 当XML格式错误时
    """
    root = read_xml(file_path)
    return xml_to_dict(root)
