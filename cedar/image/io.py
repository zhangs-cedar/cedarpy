import cv2
import base64

from PIL import Image
from io import BytesIO
from urllib.parse import quote


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