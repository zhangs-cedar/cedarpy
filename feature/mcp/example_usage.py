#!/usr/bin/env python3
"""OpenCV MCP服务使用示例

这个文件展示了如何在Cursor中使用OpenCV MCP服务进行图像处理
"""

import os
import cv2
import numpy as np
from smart_opencv_mcp import process_image_by_description


def create_sample_image() -> str:
    """创建示例图像用于测试
    
    Returns:
        示例图像的路径
    """
    # 创建一个包含多种图形的测试图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 添加一些几何图形
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)  # 绿色矩形
    cv2.circle(img, (300, 100), 50, (255, 0, 0), -1)  # 蓝色圆形
    cv2.ellipse(img, (200, 250), (80, 40), 45, 0, 360, (0, 0, 255), -1)  # 红色椭圆
    
    # 添加一些噪声
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    output_path = "sample_image.jpg"
    cv2.imwrite(output_path, img)
    print(f"✅ 创建示例图像: {output_path}")
    
    return output_path


def example_basic_processing():
    """示例1: 基础图像处理"""
    print("\n🔄 示例1: 基础图像处理")
    
    sample_image = create_sample_image()
    
    # 基础边缘检测
    description = "读取图片，进行高斯模糊，然后进行边缘检测"
    
    result = process_image_by_description(
        description=description,
        input_path=sample_image,
        output_dir="output"
    )
    
    print(f"处理状态: {result['status']}")
    print(f"输出文件: {result['final_output']}")
    
    # 清理示例文件
    os.remove(sample_image)


def example_advanced_processing():
    """示例2: 高级图像处理流程"""
    print("\n🔄 示例2: 高级图像处理流程")
    
    sample_image = create_sample_image()
    
    # 复杂的处理流程
    description = """
    读取图片，
    进行中值滤波(核大小:5)降噪，
    然后进行阈值处理(阈值:127,最大值:255)，
    最后进行形态学操作(操作:open,核大小:3)优化结果
    """
    
    result = process_image_by_description(
        description=description,
        input_path=sample_image,
        output_dir="output"
    )
    
    print(f"处理状态: {result['status']}")
    print("处理步骤:")
    for step in result['operations']:
        print(f"  步骤{step['step']}: {step['operation']} - 参数: {step['params']}")
    print(f"最终输出: {result['final_output']}")
    
    # 清理示例文件
    os.remove(sample_image)


def example_custom_parameters():
    """示例3: 自定义参数处理"""
    print("\n🔄 示例3: 自定义参数处理")
    
    sample_image = create_sample_image()
    
    # 带精确参数的处理
    description = """
    读取图片，
    进行高斯模糊(核大小:9)，
    然后进行边缘检测(阈值1:50,阈值2:150)
    """
    
    result = process_image_by_description(
        description=description,
        input_path=sample_image,
        output_dir="output"
    )
    
    print(f"处理状态: {result['status']}")
    print(f"输出文件: {result['final_output']}")
    
    # 清理示例文件
    os.remove(sample_image)


def main():
    """主函数 - 运行所有示例"""
    print("🎯 OpenCV MCP服务使用示例")
    print("=" * 50)
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    try:
        # 运行示例
        example_basic_processing()
        example_advanced_processing() 
        example_custom_parameters()
        
        print("\n✅ 所有示例运行完成!")
        print("📁 查看 output/ 目录中的处理结果")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        print("请确保MCP服务正在运行：python smart_opencv_mcp.py")


if __name__ == "__main__":
    main() 