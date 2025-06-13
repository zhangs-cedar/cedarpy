#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cedar库功能分析脚本
解析本地cedar包的模块结构和主要功能
"""

import sys
import os
import inspect
from typing import Dict, List, Any
import importlib.util

# 添加cedar包路径
cedar_path = os.path.abspath("../../cedar")
sys.path.insert(0, os.path.dirname(cedar_path))

try:
    import cedar
except ImportError as e:
    print(f"导入cedar失败: {e}")
    sys.exit(1)


def analyze_module_functions(module: Any, module_name: str) -> Dict[str, List[str]]:
    """分析模块中的函数和类
    
    Args:
        module: 要分析的模块
        module_name: 模块名称
        
    Returns:
        包含函数和类信息的字典
    """
    functions = []
    classes = []
    
    for name, obj in inspect.getmembers(module):
        if not name.startswith('_'):
            if inspect.isfunction(obj):
                functions.append(name)
            elif inspect.isclass(obj):
                classes.append(name)
    
    return {
        'functions': functions,
        'classes': classes
    }


def get_function_signature(module: Any, func_name: str) -> str:
    """获取函数签名
    
    Args:
        module: 模块对象
        func_name: 函数名
        
    Returns:
        函数签名字符串
    """
    try:
        func = getattr(module, func_name)
        return str(inspect.signature(func))
    except Exception:
        return "签名获取失败"


def analyze_cedar_structure() -> None:
    """分析cedar库的整体结构"""
    print("🌲 Cedar库结构分析")
    print("=" * 60)
    
    # 获取cedar的主要模块
    cedar_modules = {
        'utils': getattr(cedar, 'utils', None),
        'image': getattr(cedar, 'image', None), 
        'draw': getattr(cedar, 'draw', None),
        'label': getattr(cedar, 'label', None),
        'pdx': getattr(cedar, 'pdx', None),
        'supper': getattr(cedar, 'supper', None),
        'feature': getattr(cedar, 'feature', None),
    }
    
    for module_name, module_obj in cedar_modules.items():
        if module_obj is None:
            print(f"\n❌ {module_name} 模块未找到")
            continue
            
        print(f"\n📦 {module_name.upper()} 模块")
        print("-" * 40)
        
        analysis = analyze_module_functions(module_obj, module_name)
        
        if analysis['functions']:
            print(f"🔧 函数 ({len(analysis['functions'])}个):")
            for func in analysis['functions']:
                signature = get_function_signature(module_obj, func)
                print(f"  • {func}{signature}")
        
        if analysis['classes']:
            print(f"🏗️  类 ({len(analysis['classes'])}个):")
            for cls in analysis['classes']:
                print(f"  • {cls}")


def analyze_key_features() -> None:
    """分析cedar库的关键功能"""
    print("\n\n🎯 Cedar库核心功能分析")
    print("=" * 60)
    
    # 1. 图像处理功能
    print("\n📸 图像处理功能 (cedar.image)")
    print("-" * 30)
    try:
        from cedar.image import imread, imwrite, calculate_iou, merge_boxes, roate_image
        from cedar.image import array_to_base64, path_to_url, url_to_path, find_image_path, is_image
        
        image_features = [
            ("imread", "图像读取，支持中文路径"),
            ("imwrite", "图像写入，支持中文路径"),
            ("calculate_iou", "计算两个边界框的IoU"),
            ("merge_boxes", "合并重叠的边界框"),
            ("roate_image", "图像旋转"),
            ("array_to_base64", "数组转Base64编码"),
            ("path_to_url", "路径转URL"),
            ("url_to_path", "URL转路径"),
            ("find_image_path", "查找图像路径"),
            ("is_image", "判断是否为图像文件"),
        ]
        
        for func_name, description in image_features:
            print(f"  ✅ {func_name}: {description}")
            
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
    
    # 2. 绘图功能
    print("\n🎨 绘图功能 (cedar.draw)")
    print("-" * 30)
    try:
        from cedar.draw import color_list, draw_lines, putText, imshow
        
        draw_features = [
            ("color_list", "颜色列表，包含常用颜色"),
            ("draw_lines", "绘制线条"),
            ("putText", "图像上添加文本（支持中文）"),
            ("imshow", "matplotlib显示图像"),
        ]
        
        for func_name, description in draw_features:
            print(f"  ✅ {func_name}: {description}")
            
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
    
    # 3. 工具功能
    print("\n🛠️  工具功能 (cedar.utils)")
    print("-" * 30)
    try:
        from cedar.utils import (Config, Dict2Obj, init_logger, rmtree_makedirs,
                               split_filename, timeit, set_timeit_env, create_name,
                               run_subprocess, get_file_md5, find_duplicate_filenames,
                               move_file, copy_file, get_files_list)
        
        utils_features = [
            ("Config", "配置管理类"),
            ("Dict2Obj", "字典转对象工具"),
            ("init_logger", "日志初始化"),
            ("rmtree_makedirs", "删除并重建目录"),
            ("split_filename", "分离文件名和扩展名"),
            ("timeit", "函数执行时间装饰器"),
            ("create_name", "创建唯一名称"),
            ("run_subprocess", "运行子进程"),
            ("get_file_md5", "获取文件MD5"),
            ("find_duplicate_filenames", "查找重复文件名"),
            ("move_file", "移动文件"),
            ("copy_file", "复制文件"),
            ("get_files_list", "获取文件列表"),
        ]
        
        for func_name, description in utils_features:
            print(f"  ✅ {func_name}: {description}")
            
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")


def demo_cedar_usage() -> None:
    """演示cedar库的基本用法"""
    print("\n\n🚀 Cedar库使用示例")
    print("=" * 60)
    
    try:
        # 演示图像处理
        print("\n📸 图像处理示例:")
        from cedar.image import imread, is_image
        
        print("  • 检查test_image.jpg是否为图像文件:")
        result = is_image("test_image.jpg")
        print(f"    结果: {result}")
        
        if result:
            print("  • 读取图像:")
            img = imread("test_image.jpg")
            print(f"    图像尺寸: {img.shape}")
        
        # 演示工具功能
        print("\n🛠️  工具功能示例:")
        from cedar.utils import split_filename, create_name, timeit
        
        print("  • 分离文件名:")
        filename, ext = split_filename("test_image.jpg")
        print(f"    文件名: {filename}, 扩展名: {ext}")
        
        print("  • 创建唯一名称:")
        unique_name = create_name("test")
        print(f"    唯一名称: {unique_name}")
        
        # 演示绘图功能
        print("\n🎨 绘图功能示例:")
        from cedar.draw import color_list
        
        print("  • 可用颜色列表(前5个):")
        for i, color in enumerate(color_list[:5]):
            print(f"    {i+1}. {color}")
            
    except Exception as e:
        print(f"  ❌ 演示过程中发生错误: {e}")


def main():
    """主函数"""
    print("🌲 Cedar本地包功能解析器")
    print("=" * 60)
    
    # 显示cedar路径信息
    print(f"Cedar包路径: {cedar_path}")
    
    try:
        # 分析库结构
        analyze_cedar_structure()
        
        # 分析关键功能
        analyze_key_features()
        
        # 演示使用方法
        demo_cedar_usage()
        
        print("\n\n✅ Cedar库分析完成！")
        print("\n📝 总结:")
        print("Cedar是一个功能丰富的Python工具库，主要包含:")
        print("  • 图像处理: 读写、IoU计算、旋转等")
        print("  • 绘图工具: 颜色管理、文本绘制、线条绘制")
        print("  • 实用工具: 文件操作、配置管理、日志等")
        print("  • 机器学习: 特征提取、可训练分割等")
        print("  • 数据处理: 标签处理、pandas扩展等")
        
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 