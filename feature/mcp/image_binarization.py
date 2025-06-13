#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像二值化处理模块
使用OpenCV对图像进行二值化操作
"""

import cv2
import numpy as np
from typing import Tuple


def load_image(image_path: str) -> np.ndarray:
    """加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        加载的BGR图像，形状(H,W,3)
        
    Raises:
        FileNotFoundError: 当图像文件不存在时
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像文件: {image_path}")
    return image


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """将彩色图像转换为灰度图
    
    Args:
        img: 输入BGR图像，形状(H,W,3)
        
    Returns:
        灰度图像，形状(H,W)
    """
    if len(img.shape) == 3:
        # 转换为灰度图（OpenCV处理规范）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray


def apply_threshold(gray_img: np.ndarray, threshold_value: int = 127, 
                   max_value: int = 255, threshold_type: int = cv2.THRESH_BINARY) -> Tuple[float, np.ndarray]:
    """应用阈值二值化
    
    Args:
        gray_img: 输入灰度图像，形状(H,W)
        threshold_value: 阈值，默认127
        max_value: 最大值，默认255
        threshold_type: 阈值类型，默认为THRESH_BINARY
        
    Returns:
        元组：(实际使用的阈值, 二值化后的图像)
    """
    ret, binary_img = cv2.threshold(gray_img, threshold_value, max_value, threshold_type)
    return ret, binary_img


def apply_adaptive_threshold(gray_img: np.ndarray, max_value: int = 255, 
                           adaptive_method: int = cv2.ADAPTIVE_THRESH_MEAN_C,
                           threshold_type: int = cv2.THRESH_BINARY,
                           block_size: int = 11, c: int = 2) -> np.ndarray:
    """应用自适应阈值二值化
    
    Args:
        gray_img: 输入灰度图像，形状(H,W)
        max_value: 最大值，默认255
        adaptive_method: 自适应方法，默认为ADAPTIVE_THRESH_MEAN_C
        threshold_type: 阈值类型，默认为THRESH_BINARY
        block_size: 邻域大小，默认11
        c: 常数项，默认2
        
    Returns:
        自适应二值化后的图像，形状(H,W)
    """
    adaptive_binary = cv2.adaptiveThreshold(
        gray_img, max_value, adaptive_method, threshold_type, block_size, c
    )
    return adaptive_binary


def apply_otsu_threshold(gray_img: np.ndarray) -> Tuple[float, np.ndarray]:
    """应用Otsu自动阈值二值化
    
    Args:
        gray_img: 输入灰度图像，形状(H,W)
        
    Returns:
        元组：(Otsu阈值, 二值化后的图像)
    """
    ret, otsu_binary = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return ret, otsu_binary


def save_image(image: np.ndarray, output_path: str) -> None:
    """保存图像到文件
    
    Args:
        image: 要保存的图像
        output_path: 输出文件路径
    """
    cv2.imwrite(output_path, image)
    print(f"图像已保存到: {output_path}")


def display_images(original: np.ndarray, binary: np.ndarray, window_name: str = "图像对比") -> None:
    """显示原图和二值化结果对比
    
    Args:
        original: 原始图像
        binary: 二值化图像
        window_name: 窗口名称
    """
    # 将图像水平拼接显示
    if len(original.shape) == 3:
        original_gray = convert_to_grayscale(original)
    else:
        original_gray = original
        
    combined = np.hstack((original_gray, binary))
    
    cv2.imshow(window_name, combined)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """主函数：对test_image.jpg进行二值化处理"""
    image_path = "test_image.jpg"
    
    try:
        # 1. 加载图像
        print("正在加载图像...")
        original_img = load_image(image_path)
        print(f"图像尺寸: {original_img.shape}")
        
        # 2. 转换为灰度图
        print("转换为灰度图...")
        gray_img = convert_to_grayscale(original_img)
        
        # 3. 固定阈值二值化
        print("应用固定阈值二值化...")
        _, binary_fixed = apply_threshold(gray_img, threshold_value=127)
        save_image(binary_fixed, "binary_fixed_threshold.jpg")
        
        # 4. 自适应阈值二值化
        print("应用自适应阈值二值化...")
        binary_adaptive = apply_adaptive_threshold(gray_img)
        save_image(binary_adaptive, "binary_adaptive_threshold.jpg")
        
        # 5. Otsu自动阈值二值化
        print("应用Otsu自动阈值二值化...")
        otsu_threshold, binary_otsu = apply_otsu_threshold(gray_img)
        print(f"Otsu阈值: {otsu_threshold:.2f}")
        save_image(binary_otsu, "binary_otsu_threshold.jpg")
        
        # 6. 显示对比结果（使用Otsu结果）
        print("显示图像对比...")
        display_images(original_img, binary_otsu, "原图 vs Otsu二值化")
        
        print("二值化处理完成！")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    main() 