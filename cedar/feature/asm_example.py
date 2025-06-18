"""ASM使用示例

该模块展示了如何使用ASM进行形状匹配。
"""

import numpy as np
import cv2
from typing import List, Tuple
from .asm import ASM, ShapeModel
from .asm_trainer import ASMTrainer


def create_synthetic_shapes(
    n_shapes: int,
    n_points: int,
    noise_level: float = 0.1
) -> List[np.ndarray]:
    """创建合成训练数据
    
    Args:
        n_shapes: 形状数量
        n_points: 每个形状的点数
        noise_level: 噪声水平
        
    Returns:
        形状列表
    """
    # 创建基础椭圆
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    base_shape = np.column_stack([
        100 + 50 * np.cos(t),
        100 + 30 * np.sin(t)
    ])
    
    shapes = []
    for _ in range(n_shapes):
        # 添加随机变形
        noise = np.random.normal(0, noise_level, base_shape.shape)
        shape = base_shape + noise
        shapes.append(shape)
    
    return shapes


def main():
    """主函数，展示ASM的使用"""
    # 创建训练数据
    n_shapes = 20
    n_points = 40
    shapes = create_synthetic_shapes(n_shapes, n_points)
    
    # 训练模型
    trainer = ASMTrainer(n_modes=5)
    for shape in shapes:
        trainer.add_shape(shape)
    shape_model = trainer.train()
    
    # 创建ASM实例
    asm = ASM(shape_model)
    
    # 创建测试图像
    image = np.zeros((300, 300), dtype=np.uint8)
    
    # 在图像上绘制一个变形的椭圆
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    test_shape = np.column_stack([
        150 + 60 * np.cos(t),
        150 + 40 * np.sin(t)
    ])
    test_shape = test_shape.astype(np.int32)
    
    # 绘制测试形状
    cv2.polylines(image, [test_shape], True, 255, 2)
    
    # 添加一些噪声
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # 使用ASM拟合形状
    fitted_shape = asm.fit(image)
    
    # 可视化结果
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = asm.draw_shape(result, fitted_shape, color=(0, 255, 0))
    
    # 显示结果
    cv2.imshow("ASM Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 