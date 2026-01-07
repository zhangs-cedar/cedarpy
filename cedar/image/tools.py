import base64
import os
import os.path as osp
from io import BytesIO
from urllib.parse import quote, unquote

import cv2
import numpy as np
from PIL import Image


# ==================== 图像读写 ====================

def imread(image_path, flag=cv2.IMREAD_COLOR):
    """读取图片，支持中文路径"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flag)
    if img is None:
        raise FileNotFoundError(f'无法读取: {image_path}')
    return img


def imwrite(image_path, img, plt=False):
    """保存图片，支持中文路径"""
    if img is None:
        raise ValueError('图片不能为空')

    if plt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs(osp.dirname(image_path), exist_ok=True)
    suffix = osp.splitext(image_path)[1]
    cv2.imencode(suffix, img)[1].tofile(image_path)


# ==================== 图像变换 ====================

def rotate_image(img, angle, border_value=(0, 0, 0)):
    """对图像进行旋转操作

    Args:
        img: 待旋转的图像
        angle: 旋转角度（度），正数逆时针，负数顺时针
        border_value: 边界填充颜色，默认黑色

    Returns:
        旋转后的图像（保持原始尺寸）
    """
    if img is None:
        raise ValueError('图像不能为空')

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])

    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)

    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy

    rotated = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=border_value)

    pw, ph = (nw - w) // 2, (nh - h) // 2
    return rotated[ph:ph + h, pw:pw + w]


# ==================== 轮廓处理 ====================

def get_contours(mask):
    """获取掩码图像的轮廓"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_minAreaRect(cnt):
    """获取轮廓的最小外接矩形"""
    return cv2.minAreaRect(cnt)


# ==================== 向量计算 ====================

def get_vector(point1, point2):
    """根据两个点计算向量（y轴向上为正）"""
    if point1[0] > point2[0]:
        return [point1[0] - point2[0], point2[1] - point1[1]]
    return [point2[0] - point1[0], point1[1] - point2[1]]


def get_longside_rect_ps(rect_ps):
    """获取矩形长边的向量"""
    p0, p1, p2 = rect_ps[0], rect_ps[1], rect_ps[-1]
    dis1 = np.linalg.norm(p0 - p1)
    dis2 = np.linalg.norm(p0 - p2)

    if dis1 > dis2:
        return get_vector(p0.tolist(), p1.tolist())
    return get_vector(p0.tolist(), p2.tolist())


def calcu_angle_between_vectors(v1, v2):
    """计算两个向量之间的夹角（度）"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    angle = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(angle)


def calcu_angle(v1):
    """计算向量与x轴正方向的夹角（度）"""
    return calcu_angle_between_vectors(v1, [1, 0])


# ==================== IoU 计算 ====================

def calculate_iou(box1, box2):
    """计算两个矩形框之间的交并比(IoU)，格式为(x, y, w, h)"""
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError('边界框必须包含4个元素: (x, y, width, height)')

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1 + w1, x2 + w2) - xi
    hi = min(y1 + h1, y2 + h2) - yi
    intersection = max(0, wi) * max(0, hi)

    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def merge_boxes(boxes, iou_threshold):
    """合并重叠的边界框"""
    if not boxes:
        return []

    if not 0 <= iou_threshold <= 1:
        raise ValueError('IoU阈值必须在[0, 1]范围内')

    merged = []
    for i, box in enumerate(boxes):
        if len(box) != 4:
            raise ValueError(f'边界框 {i} 必须包含4个元素: [x, y, width, height]')

        overlapped = False
        for j, mb in enumerate(merged):
            if calculate_iou(box, mb) > iou_threshold:
                x = min(box[0], mb[0])
                y = min(box[1], mb[1])
                w = max(box[0] + box[2], mb[0] + mb[2]) - x
                h = max(box[1] + box[3], mb[1] + mb[3]) - y
                merged[j] = [x, y, w, h]
                overlapped = True
                break

        if not overlapped:
            merged.append(box)

    return merged


# ==================== 路径/格式转换 ====================

def find_image_path(file_path, name, extensions=None):
    """在给定文件路径的同一目录下查找图片文件"""
    if extensions is None:
        extensions = ['.png', '.bmp', '.jpg']
    dir_path = osp.dirname(file_path)
    for ext in extensions:
        img_path = osp.join(dir_path, f'{name}{ext}')
        if osp.exists(img_path):
            return img_path
    return None


def is_image(file_path):
    """判断文件是否为有效的图像文件"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    return osp.splitext(file_path)[1].lower() in valid_extensions


def array_to_base64(image_array):
    """将NumPy图像数组转换为Base64编码的PNG图像字符串"""
    if image_array is None:
        raise ValueError('图像数组不能为空')

    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img_bgr)
    buffered = BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'


def path_to_url(file_path):
    """将本地路径转换为file URL"""
    if not file_path:
        raise ValueError('文件路径不能为空')

    encoded_path = quote(file_path.replace('\\\\', '/')).replace('%3A', ':').replace('%5C', '/')
    url = f'file:///{encoded_path}'
    return url.replace('%252F', '%2F')


def url_to_path(encoded_url):
    """将编码后的URL转换回本地文件路径"""
    if not encoded_url:
        raise ValueError('URL不能为空')

    if not encoded_url.startswith('file:///'):
        raise ValueError('提供的 URL 不是有效的 file URL')

    decoded_path = unquote(encoded_url[8:])
    return decoded_path.replace('/', '\\')
