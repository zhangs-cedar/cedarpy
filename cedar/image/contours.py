
"""
mask : 0/255 mask of the image, 0 for the background, 255 for the foreground
"""

import cv2
import numpy as np

def get_contours(mask):
    """
    Get the contours of a mask.
    Parameters:
        mask : 0/255 mask of the image, 0 for the background, 255 for the foreground
    Returns:
        contours : list of tuples (x, y) representing the contour points
    """
    # Find the contours in the mask
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_vertios(point1,point2):
    """
    根据两个点计算向量。向量的坐标系为直角坐标系，x轴为水平方向，y轴为垂直方向。
    注：y需要取负值，因为y轴向上为正。
    
    Args:
        point1 (list): 第一个点的坐标，格式为 [x, y]。
        point2 (list): 第二个点的坐标，格式为 [x, y]。
    
    Returns:
        list: 两个点的向量，格式为 [x, y]。
    
    """
    
    if point1[0]>point2[0]:
        v = [point1[0]-point2[0],point2[1]-point1[1]]
    else:
        v = [point2[0]-point1[0],point1[1]-point2[1]]
    return v
    
def get_longside_rect_ps(rect_ps):
    """  """
    p0 = rect_ps[0]
    p1 = rect_ps[1]
    p2 = rect_ps[-1]
    dis1 = np.linalg.norm((p0-p1))
    dis2 = np.linalg.norm((p0-p2))
    if dis1>dis2:
        return get_vertios(p0,p1)
    else:
        return get_vertios(p0,p2)
    
def calcu_angle_between_verctors(v1,v2):
    """
    计算两个向量之间的夹角。(0-180度)
    """
    # 计算向量的点积
    dot_product = np.dot(v1, v2)
    # 计算向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 计算夹角的弧度值
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))
    # 将弧度值转换为角度值
    angle_deg = np.degrees(angle)
    return angle_deg
    
def calcu_angle(v1):
    v2 = [1,0]
    angle_deg = calcu_angle_between_verctors(v1, v2)
    return angle_deg

def get_minAreaRect(cnt):
    """
    Get the minimum area rectangle of a contour.
    Parameters:
        cnt : contour
    Returns:
        minAreaRect : tuple (x, y, width, height) representing the minimum area rectangle
    """
    # Find the minimum area rectangle of the contour
    rect = cv2.minAreaRect(cnt)
    (center_x, center_y), (width, height), _angle = rect
    rect_ps = cv2.boxPoints(rect)
    box = np.int0(rect_ps)
    return rect

