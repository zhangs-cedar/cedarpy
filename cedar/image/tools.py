import cv2
import numpy as np


def roate_image(img, angle, borderValue=(0, 0, 0)):
    """
    对图像进行旋转操作, 返回旋转后的图像.
    注：目前仅支持正方形图像的旋转操作，且旋转角度为正数表示逆时针旋转，负数表示顺时针旋转。

    Args:
        img (numpy.ndarray): 待旋转的图像，应为numpy数组类型。
        angle (float): 旋转角度，单位为度，正数表示逆时针旋转，负数表示顺时针旋转。
        borderValue (tuple, optional): 旋转后图像边界外的填充颜色，默认为黑色(0, 0, 0)。填充颜色应为包含三个整数的元组，分别代表BGR颜色通道的值。
    Returns:
        numpy.ndarray: 旋转后的图像，返回numpy数组类型。

    """
    h, w = img.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    img = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_NEAREST, borderValue=borderValue)
    p_w = (nW - w) // 2
    p_h = (nH - h) // 2

    return img[p_h : p_h + h, p_w : p_w + w]
