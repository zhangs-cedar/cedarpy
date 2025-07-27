#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
颜色打印测试脚本
展示不同类型数据的颜色输出效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s_print import print

def test_basic_types():
    """测试基本数据类型"""
    print("=== 基本数据类型测试 ===")
    print("字符串:", "Hello World")
    print("整数:", 42)
    print("浮点数:", 3.14159)
    print("布尔值:", True, False)
    print("None值:", None)

def test_collections():
    """测试集合类型"""
    print("\n=== 集合类型测试 ===")
    print("列表:", [1, 2, 3, "hello"])
    print("元组:", (1, 2, 3, "world"))
    print("字典:", {"name": "张三", "age": 25})
    print("集合:", {1, 2, 3, 4, 5})

def test_functions_and_classes():
    """测试函数和类"""
    print("\n=== 函数和类测试 ===")
    
    def test_function():
        return "这是一个测试函数"
    
    class TestClass:
        def __init__(self):
            self.name = "测试类"
    
    print("函数对象:", test_function)
    print("类对象:", TestClass)
    print("类实例:", TestClass())

def test_complex_data():
    """测试复杂数据结构"""
    print("\n=== 复杂数据结构测试 ===")
    
    # 嵌套数据结构
    complex_data = {
        "users": [
            {"name": "张三", "age": 25, "skills": ["Python", "Java"]},
            {"name": "李四", "age": 30, "skills": ["C++", "Go"]}
        ],
        "settings": {
            "debug": True,
            "version": "1.0.0",
            "features": {"color": True, "logging": True}
        }
    }
    
    print("复杂嵌套数据:", complex_data)

if __name__ == "__main__":
    print("开始颜色打印测试...")
    
    test_basic_types()
    test_collections()
    test_functions_and_classes()
    test_complex_data()
    
    print("\n测试完成！") 