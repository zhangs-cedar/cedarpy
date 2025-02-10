import cv2
import base64
import os.path as osp
from PIL import Image
from io import BytesIO
from urllib.parse import quote, unquote


def find_image_path(file_path, name, extensions=[".png", ".bmp", ".jpg"]):
    """
    在给定文件路径的同一目录下，根据文件名和扩展名列表查找图片文件的完整路径。
    """
    for ext in extensions:
        img_path = osp.join(osp.dirname(file_path), f"{name}{ext}")
        if osp.exists(img_path):
            return img_path
    return None


def is_image(file_path):
    """
    判断给定的文件路径是否指向一个有效且支持的图像文件。
    """
    valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    file_extension = osp.splitext(file_path)[1].lower()
    return file_extension in valid_extensions


def array_to_base64(image_array):
    """
    将NumPy图像数组转换为Base64编码的PNG图像字符串。

    Args:
        image_array (np.ndarray): 形状为 (height, width, channels) 的NumPy图像数组。

    Returns:
        str: 包含Base64编码的PNG图像数据的字符串，格式为 "data:image/png;base64,{img_str}"。

    """
    # 创建PIL图像
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(image_array)
    # 保存图像到字节流
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    # 将字节流转换为Base64编码的字符串
    img_str = base64.b64encode(buffered.getvalue()).decode()
    # 返回Base64图像字符串
    return f"data:image/png;base64,{img_str}"


# 定义转换 Windows 路径到 file URL 的方法
def path_to_url(file_path):
    # 将本地路径转换为 URL 编码的字符串，但不编码驱动器号
    encoded_path = quote(file_path.replace("\\\\", "/")).replace("%3A", ":").replace("%5C", "/")
    # 构建 file URL
    url = f"file:///{encoded_path}"
    # 修复任何可能的编码问题，如将 %252F 转换为单个 /
    url = url.replace("%252F", "%2F")
    return url


def url_to_path(encoded_url):
    """
    将编码后的 URL 转换回本地文件路径。
    """
    # 从 URL 中移除 file:/// 前缀
    if encoded_url.startswith("file:///"):
        encoded_path = encoded_url[8:]
    else:
        raise ValueError("提供的 URL 不是有效的 file URL")
    decoded_path = unquote(encoded_path)
    local_path = decoded_path.replace("/", "\\")
    return local_path
