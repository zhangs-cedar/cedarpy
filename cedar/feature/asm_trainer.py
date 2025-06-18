"""ASM模型训练器

该模块提供了用于训练ASM模型的工具类。
"""

import numpy as np
from typing import List, Tuple
from .asm import ShapeModel


class ASMTrainer:
    """ASM模型训练器
    
    用于从训练数据构建形状模型。
    
    Attributes:
        shapes: 训练形状列表
        n_points: 每个形状的点数
        n_modes: 使用的模式数量
    """
    
    def __init__(self, n_modes: int = 10):
        """初始化训练器
        
        Args:
            n_modes: 使用的模式数量
        """
        self.shapes: List[np.ndarray] = []
        self.n_points: int = 0
        self.n_modes = n_modes

    def add_shape(self, shape: np.ndarray) -> None:
        """添加训练形状
        
        Args:
            shape: 形状点集，形状为(n_points, 2)
        """
        if self.n_points == 0:
            self.n_points = len(shape)
        elif len(shape) != self.n_points:
            raise ValueError(f"形状点数不匹配: 期望 {self.n_points}, 实际 {len(shape)}")
        
        self.shapes.append(shape)

    def _align_shapes(self) -> np.ndarray:
        """对齐所有训练形状
        
        Returns:
            对齐后的形状数组，形状为(n_shapes, n_points, 2)
        """
        if not self.shapes:
            raise ValueError("没有训练形状")
        
        # 使用第一个形状作为参考
        reference_shape = self.shapes[0]
        aligned_shapes = [reference_shape]
        
        for shape in self.shapes[1:]:
            # 计算质心
            ref_center = np.mean(reference_shape, axis=0)
            shape_center = np.mean(shape, axis=0)
            
            # 去中心化
            ref_centered = reference_shape - ref_center
            shape_centered = shape - shape_center
            
            # 计算缩放因子
            ref_scale = np.sqrt(np.sum(ref_centered ** 2))
            shape_scale = np.sqrt(np.sum(shape_centered ** 2))
            scale = ref_scale / shape_scale
            
            # 计算旋转矩阵
            H = shape_centered.T @ ref_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # 应用变换
            aligned_shape = scale * (R @ shape.T).T + ref_center
            aligned_shapes.append(aligned_shape)
        
        return np.array(aligned_shapes)

    def train(self) -> ShapeModel:
        """训练ASM模型
        
        Returns:
            训练好的形状模型
        """
        if not self.shapes:
            raise ValueError("没有训练形状")
        
        # 对齐形状
        aligned_shapes = self._align_shapes()
        
        # 计算平均形状
        mean_shape = np.mean(aligned_shapes, axis=0)
        
        # 计算协方差矩阵
        n_shapes = len(self.shapes)
        shape_vectors = aligned_shapes.reshape(n_shapes, -1)
        mean_vector = mean_shape.flatten()
        centered_vectors = shape_vectors - mean_vector
        covariance = centered_vectors.T @ centered_vectors / (n_shapes - 1)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # 按特征值降序排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 选择前n_modes个模式
        eigenvalues = eigenvalues[:self.n_modes]
        eigenvectors = eigenvectors[:, :self.n_modes]
        
        return ShapeModel(
            mean_shape=mean_shape,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            n_modes=self.n_modes
        ) 