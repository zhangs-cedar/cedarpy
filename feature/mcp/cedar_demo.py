#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cedar库完整功能演示脚本
展示cedar库各个模块的具体使用方法
"""

import sys
import os
import numpy as np

# 添加cedar包路径
cedar_path = os.path.abspath("../../cedar")
sys.path.insert(0, os.path.dirname(cedar_path))

try:
    import cedar
except ImportError as e:
    print(f"导入cedar失败: {e}")
    sys.exit(1)


def demo_image_processing():
    """演示图像处理功能"""
    print("\n🖼️  图像处理功能演示")
    print("=" * 50)
    
    from cedar.image import imread, imwrite, is_image, calculate_iou, array_to_base64
    
    # 1. 图像读取与检查
    print("1. 图像文件检查与读取:")
    image_path = "test_image.jpg"
    
    if is_image(image_path):
        print(f"  ✅ {image_path} 是有效的图像文件")
        
        # 读取图像
        img = imread(image_path)
        print(f"  📏 图像尺寸: {img.shape}")
        print(f"  📊 数据类型: {img.dtype}")
        
        # 转换为base64
        base64_str = array_to_base64(img)
        print(f"  🔗 Base64编码长度: {len(base64_str)} 字符")
        
    # 2. IoU计算演示
    print("\n2. 边界框IoU计算:")
    box1 = [50, 50, 150, 150]  # [x1, y1, x2, y2]
    box2 = [100, 100, 200, 200]
    
    iou = calculate_iou(box1, box2)
    print(f"  📦 Box1: {box1}")
    print(f"  📦 Box2: {box2}")
    print(f"  🎯 IoU值: {iou:.4f}")


def demo_drawing_features():
    """演示绘图功能"""
    print("\n🎨 绘图功能演示")
    print("=" * 50)
    
    from cedar.image import imread, imwrite
    from cedar.draw import putText, color_list, draw_lines
    
    # 读取原图
    img = imread("test_image.jpg")
    
    # 1. 添加中文文本
    print("1. 添加中文文本:")
    img_with_text = putText(
        img.copy(), 
        "Cedar库演示 🌲", 
        position=(50, 50), 
        textColor=(0, 255, 0), 
        textSize=40
    )
    
    # 添加英文文本
    img_with_text = putText(
        img_with_text, 
        "Image Processing Demo", 
        position=(50, 100), 
        textColor=(255, 0, 0), 
        textSize=30
    )
    
    # 保存结果
    imwrite("cedar_text_demo.jpg", img_with_text)
    print(f"  ✅ 已保存带文本的图像: cedar_text_demo.jpg")
    
    # 2. 颜色列表展示
    print("\n2. 可用颜色列表(前10个):")
    for i, color in enumerate(color_list[:10]):
        print(f"  🎨 颜色{i+1}: {color}")
    
    # 3. 绘制线条
    print("\n3. 绘制线条:")
    img_with_lines = img.copy()
    
    # 绘制网格线 (row_lines 水平线, col_lines 垂直线)
    row_lines = [50, 100, 150, 200]  # 水平线的y坐标
    col_lines = [50, 100, 150, 200]  # 垂直线的x坐标
    
    img_with_lines = draw_lines(img_with_lines, row_lines, col_lines, color=(0, 255, 255), thickness=2)
    imwrite("cedar_lines_demo.jpg", img_with_lines)
    print(f"  ✅ 已保存带网格线的图像: cedar_lines_demo.jpg")


def demo_utility_functions():
    """演示工具功能"""
    print("\n🛠️  工具功能演示")
    print("=" * 50)
    
    from cedar.utils import (split_filename, create_name, get_file_md5, 
                            rmtree_makedirs, timeit, Config, Dict2Obj)
    
    # 1. 文件名处理
    print("1. 文件名处理:")
    filename, ext = split_filename("test_image.jpg")
    print(f"  📄 原文件名: test_image.jpg")
    print(f"  📝 文件名: {filename}")
    print(f"  🏷️  扩展名: {ext}")
    
    # 2. 创建唯一名称
    print("\n2. 创建唯一名称:")
    unique_name = create_name()
    print(f"  🆔 唯一名称: {unique_name}")
    
    # 3. 文件MD5计算
    print("\n3. 文件MD5计算:")
    if os.path.exists("test_image.jpg"):
        md5_hash = get_file_md5("test_image.jpg")
        print(f"  🔐 test_image.jpg的MD5: {md5_hash}")
    
    # 4. 字典转对象
    print("\n4. 字典转对象:")
    test_dict = {"name": "Cedar", "version": "1.0", "features": ["image", "draw", "utils"]}
    obj = Dict2Obj(test_dict)
    print(f"  📦 原字典: {test_dict}")
    print(f"  🔧 对象访问: obj.name = {obj.name}, obj.version = {obj.version}")
    
    # 5. 配置管理
    print("\n5. 配置管理演示:")
    # 创建一个临时配置文件
    import json
    config_data = {
        "image_size": [300, 300],
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    # 保存临时配置文件
    with open("temp_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # 使用Config类加载配置
    config = Config("temp_config.json")
    print(f"  ⚙️  配置对象: image_size={config.obj.image_size}, batch_size={config.obj.batch_size}")
    
    # 清理临时文件
    os.remove("temp_config.json")


def demo_advanced_features():
    """演示高级功能"""
    print("\n🚀 高级功能演示")
    print("=" * 50)
    
    # 1. 时间装饰器演示
    print("1. 时间装饰器演示:")
    from cedar.utils import timeit
    
    @timeit
    def slow_function():
        """模拟耗时操作"""
        import time
        time.sleep(0.1)
        return "处理完成"
    
    result = slow_function()
    print(f"  ⏱️  函数执行结果: {result}")
    
    # 2. 图像处理综合示例
    print("\n2. 图像处理综合示例:")
    from cedar.image import imread, imwrite
    from cedar.draw import putText, color_list
    
    # 读取原图
    img = imread("test_image.jpg")
    
    # 创建综合处理图像
    processed_img = img.copy()
    
    # 添加标题
    processed_img = putText(
        processed_img, 
        "🌲 Cedar库功能展示", 
        position=(20, 30), 
        textColor=tuple(color_list[1]), 
        textSize=35
    )
    
    # 添加功能说明
    features = [
        "✓ 图像读写 (支持中文路径)",
        "✓ 文本绘制 (支持中文)",
        "✓ 工具函数集合",
        "✓ 配置管理",
        "✓ 时间装饰器"
    ]
    
    for i, feature in enumerate(features):
        y_pos = 80 + i * 35
        processed_img = putText(
            processed_img, 
            feature, 
            position=(20, y_pos), 
            textColor=tuple(color_list[i % len(color_list)]), 
            textSize=20
        )
    
    # 保存最终结果
    imwrite("cedar_comprehensive_demo.jpg", processed_img)
    print(f"  ✅ 已保存综合演示图像: cedar_comprehensive_demo.jpg")


def main():
    """主函数"""
    print("🌲 Cedar库完整功能演示")
    print("=" * 60)
    
    try:
        # 演示各个功能模块
        demo_image_processing()
        demo_drawing_features()
        demo_utility_functions()
        demo_advanced_features()
        
        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("\n📋 生成的文件:")
        generated_files = [
            "cedar_text_demo.jpg",
            "cedar_lines_demo.jpg", 
            "cedar_comprehensive_demo.jpg"
        ]
        
        for file in generated_files:
            if os.path.exists(file):
                print(f"  📁 {file}")
        
        print("\n📖 Cedar库主要特点:")
        print("  🎯 专注于图像处理和计算机视觉")
        print("  🌍 完美支持中文路径和文本")
        print("  🛠️  丰富的工具函数集合")
        print("  ⚡ 高效的图像I/O操作")
        print("  🎨 便捷的图像绘制功能")
        print("  📊 IoU计算等CV算法")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 