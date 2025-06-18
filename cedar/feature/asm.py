"""Active Shape Model (ASM) 实现

该模块实现了可变形状匹配算法(ASM)，用于在图像中定位和匹配特定形状。
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass
class ShapeModel:
    """形状模型类
    
    存储ASM所需的形状模型参数，包括平均形状、主成分和特征值。
    
    Attributes:
        mean_shape: 平均形状点集，形状为(n_points, 2)
        eigenvectors: 主成分向量，形状为(n_points*2, n_modes)
        eigenvalues: 特征值，形状为(n_modes,)
        n_modes: 使用的模式数量
    """
    mean_shape: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    n_modes: int

    def __post_init__(self):
        """初始化后处理，确保数据类型正确"""
        self.mean_shape = np.asarray(self.mean_shape, dtype=np.float32)
        self.eigenvectors = np.asarray(self.eigenvectors, dtype=np.float32)
        self.eigenvalues = np.asarray(self.eigenvalues, dtype=np.float32)


class ASM:
    """Active Shape Model 实现
    
    实现了可变形状匹配算法，用于在图像中定位和匹配特定形状。
    
    Attributes:
        shape_model: 形状模型
        max_iterations: 最大迭代次数
        convergence_threshold: 收敛阈值
    """
    
    def __init__(
        self,
        shape_model: ShapeModel,
        max_iterations: int = 50,
        convergence_threshold: float = 0.001
    ):
        """初始化ASM
        
        Args:
            shape_model: 形状模型
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
        """
        self.shape_model = shape_model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def _align_shapes(
        self,
        shape1: np.ndarray,
        shape2: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """对齐两个形状
        
        Args:
            shape1: 第一个形状点集
            shape2: 第二个形状点集
            
        Returns:
            对齐后的形状、缩放因子、旋转角度和平移向量
        """
        # 计算质心
        center1 = np.mean(shape1, axis=0)
        center2 = np.mean(shape2, axis=0)
        
        # 去中心化
        shape1_centered = shape1 - center1
        shape2_centered = shape2 - center2
        
        # 计算缩放因子
        scale1 = np.sqrt(np.sum(shape1_centered ** 2))
        scale2 = np.sqrt(np.sum(shape2_centered ** 2))
        scale = scale2 / scale1
        
        # 计算旋转矩阵
        H = shape1_centered.T @ shape2_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 计算旋转角度
        angle = np.arctan2(R[1, 0], R[0, 0])
        
        # 计算平移向量
        translation = center2 - scale * (R @ center1)
        
        # 应用变换
        aligned_shape = scale * (R @ shape1.T).T + translation
        
        return aligned_shape, scale, angle, translation

    def _update_shape(
        self,
        current_shape: np.ndarray,
        image: np.ndarray
    ) -> np.ndarray:
        """更新形状位置
        
        Args:
            current_shape: 当前形状点集
            image: 输入图像
            
        Returns:
            更新后的形状点集
        """
        # 对每个点进行局部搜索
        updated_points = []
        for point in current_shape:
            x, y = point.astype(int)
            
            # 在点周围搜索最佳匹配位置
            search_radius = 5
            best_x, best_y = x, y
            best_response = float('-inf')
            
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                        # 使用图像梯度作为响应
                        response = np.sum(image[ny, nx])
                        if response > best_response:
                            best_response = response
                            best_x, best_y = nx, ny
            
            updated_points.append([best_x, best_y])
        
        return np.array(updated_points)

    def _constrain_shape(self, shape: np.ndarray) -> np.ndarray:
        """约束形状在模型允许的范围内
        
        Args:
            shape: 输入形状点集
            
        Returns:
            约束后的形状点集
        """
        # 将形状转换为向量
        shape_vector = shape.flatten()
        
        # 计算与平均形状的偏差
        deviation = shape_vector - self.shape_model.mean_shape.flatten()
        
        # 投影到主成分空间
        b = self.shape_model.eigenvectors.T @ deviation
        
        # 限制参数在合理范围内
        b = np.clip(b, -3 * np.sqrt(self.shape_model.eigenvalues),
                   3 * np.sqrt(self.shape_model.eigenvalues))
        
        # 重建形状
        constrained_shape = self.shape_model.mean_shape.flatten() + \
            self.shape_model.eigenvectors @ b
        
        return constrained_shape.reshape(-1, 2)

    def fit(
        self,
        image: np.ndarray,
        initial_shape: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """在图像中拟合形状
        
        Args:
            image: 输入图像
            initial_shape: 初始形状点集，如果为None则使用平均形状
            
        Returns:
            拟合后的形状点集
        """
        if initial_shape is None:
            current_shape = self.shape_model.mean_shape.copy()
        else:
            current_shape = initial_shape.copy()
        
        prev_shape = current_shape.copy()
        
        for _ in range(self.max_iterations):
            # 更新形状位置
            current_shape = self._update_shape(current_shape, image)
            
            # 约束形状
            current_shape = self._constrain_shape(current_shape)
            
            # 检查收敛
            if np.mean(np.abs(current_shape - prev_shape)) < self.convergence_threshold:
                break
                
            prev_shape = current_shape.copy()
        
        return current_shape

    def draw_shape(
        self,
        image: np.ndarray,
        shape: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """在图像上绘制形状
        
        Args:
            image: 输入图像
            shape: 形状点集
            color: 绘制颜色 (B,G,R)
            thickness: 线条粗细
            
        Returns:
            绘制了形状的图像
        """
        result = image.copy()
        points = shape.astype(np.int32)
        
        # 绘制点
        for point in points:
            cv2.circle(result, tuple(point), 2, color, -1)
        
        # 绘制连线
        for i in range(len(points)):
            pt1 = tuple(points[i])
            pt2 = tuple(points[(i + 1) % len(points)])
            cv2.line(result, pt1, pt2, color, thickness)
        
        return result 