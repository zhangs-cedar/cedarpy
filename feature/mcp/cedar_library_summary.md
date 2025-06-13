# 🌲 Cedar本地包功能解析总结

## 📖 概述

Cedar是一个专注于图像处理和计算机视觉的Python工具库，提供了丰富的功能模块，特别针对中文环境进行了优化。

## 🏗️ 库结构

Cedar库主要包含以下模块：

```
cedar/
├── image/          # 图像处理模块
├── draw/           # 绘图功能模块  
├── utils/          # 工具函数模块
├── feature/        # 特征提取模块
├── label/          # 标签处理模块
├── pdx/           # Pandas扩展模块
└── supper/        # 超级功能模块
```

## 🖼️ 图像处理模块 (cedar.image)

### 核心功能

#### 📄 图像I/O操作
- **`imread(image_path, flag=cv2.IMREAD_COLOR)`**: 读取图像，支持中文路径
- **`imwrite(image_path, img, plt=False)`**: 保存图像，支持中文路径
- **`is_image(path)`**: 判断文件是否为图像格式

#### 📐 几何计算
- **`calculate_iou(box1, box2)`**: 计算两个边界框的IoU值
- **`merge_boxes(boxes)`**: 合并重叠的边界框
- **`roate_image(img, angle)`**: 图像旋转

#### 🔄 格式转换
- **`array_to_base64(img)`**: 数组转Base64编码
- **`path_to_url(path)`**: 路径转URL格式
- **`url_to_path(url)`**: URL转路径格式
- **`find_image_path(directory)`**: 查找目录中的图像文件

### 使用示例

```python
from cedar.image import imread, imwrite, calculate_iou, is_image

# 图像读取（支持中文路径）
img = imread("测试图片.jpg")

# IoU计算
box1 = [50, 50, 150, 150]
box2 = [100, 100, 200, 200]
iou = calculate_iou(box1, box2)
print(f"IoU值: {iou:.4f}")

# 图像检查
if is_image("test.jpg"):
    print("这是一个有效的图像文件")
```

## 🎨 绘图模块 (cedar.draw)

### 核心功能

#### 🎨 颜色管理
- **`color_list`**: 预定义的颜色列表，包含多种常用颜色

#### ✏️ 文本绘制
- **`putText(img, text, position, textColor, textSize)`**: 在图像上绘制文本，支持中文

#### 📏 线条绘制
- **`draw_lines(img, row_lines, col_lines, color, thickness)`**: 绘制网格线条

#### 📊 显示功能
- **`imshow(img)`**: 使用matplotlib显示图像

### 使用示例

```python
from cedar.draw import putText, color_list, draw_lines
from cedar.image import imread

img = imread("test.jpg")

# 添加中文文本
img_with_text = putText(
    img, 
    "Cedar库演示 🌲", 
    position=(50, 50), 
    textColor=(0, 255, 0), 
    textSize=40
)

# 绘制网格线
row_lines = [50, 100, 150, 200]
col_lines = [50, 100, 150, 200]
img_with_grid = draw_lines(img, row_lines, col_lines, color=(0, 255, 255))
```

## 🛠️ 工具模块 (cedar.utils)

### 核心功能

#### 📁 文件操作
- **`split_filename(filename)`**: 分离文件名和扩展名
- **`move_file(src, dst, filename=None)`**: 移动文件
- **`copy_file(src, dst, filename=None)`**: 复制文件
- **`get_files_list(path, find_suffix=[], sortby="name")`**: 获取文件列表
- **`find_duplicate_filenames(directory)`**: 查找重复文件名

#### 🔐 安全功能
- **`get_file_md5(filename)`**: 计算文件MD5哈希值

#### ⏱️ 性能监控
- **`timeit`**: 函数执行时间装饰器
- **`set_timeit_env(debug=True)`**: 设置时间监控环境

#### 🏗️ 目录管理
- **`rmtree_makedirs(*dirs)`**: 删除并重建目录

#### 🆔 唯一标识
- **`create_name()`**: 创建基于时间戳的唯一名称

#### ⚙️ 配置管理
- **`Config(config_path)`**: 配置文件管理类，支持JSON/YAML
- **`Dict2Obj(dict_data)`**: 字典转对象工具

#### 🔄 进程管理
- **`run_subprocess(cmd, cwd=None)`**: 运行子进程

#### 📝 日志功能
- **`init_logger()`**: 初始化日志系统

### 使用示例

```python
from cedar.utils import (split_filename, create_name, get_file_md5, 
                        timeit, Config, Dict2Obj)

# 文件名处理
filename, ext = split_filename("test_image.jpg")
print(f"文件名: {filename}, 扩展名: {ext}")

# 创建唯一名称
unique_name = create_name()
print(f"唯一名称: {unique_name}")

# 时间装饰器
@timeit
def slow_function():
    import time
    time.sleep(0.1)
    return "处理完成"

# 配置管理
config = Config("config.json")
print(f"配置: {config.obj.setting_name}")

# 字典转对象
obj = Dict2Obj({"name": "Cedar", "version": "1.0"})
print(f"访问: {obj.name}")
```

## 🚀 高级特性

### 1. 中文支持
- **完美支持中文路径**: 图像读写函数专门处理中文路径编码问题
- **中文文本绘制**: putText函数支持中文字体渲染

### 2. 性能优化
- **时间装饰器**: 自动监控函数执行时间
- **批处理支持**: 文件操作支持批量处理

### 3. 配置管理
- **多格式支持**: 支持JSON、YAML配置文件
- **对象化访问**: 配置项可以像对象属性一样访问

### 4. 安全特性
- **MD5校验**: 文件完整性验证
- **安全的子进程**: 避免shell注入风险

## 📊 实际测试结果

通过演示脚本测试的功能：

### ✅ 成功测试的功能

1. **图像处理**:
   - 图像读取: 300×300×3 图像成功读取
   - IoU计算: Box1=[50,50,150,150], Box2=[100,100,200,200], IoU=0.1905
   - Base64转换: 353,990字符长度

2. **绘图功能**:
   - 中文文本绘制: "Cedar库演示 🌲"
   - 英文文本绘制: "Image Processing Demo"
   - 网格线绘制: 4×4网格

3. **工具功能**:
   - 文件名分离: "test_image.jpg" → ("test_image", ".jpg")
   - 唯一名称生成: "2025-06-11_23-28-25-924113"
   - MD5计算: "02cb88b77b73d548438c873c8c8d1878"
   - 配置管理: JSON配置文件加载成功

4. **性能监控**:
   - 时间装饰器: 0.105秒执行时间监控

### 📁 生成的演示文件

- `cedar_text_demo.jpg`: 带中英文文本的图像
- `cedar_lines_demo.jpg`: 带网格线的图像  
- `cedar_comprehensive_demo.jpg`: 综合功能展示图像

## 🎯 主要优势

1. **🌍 国际化友好**: 完美支持中文路径和文本
2. **🎨 图像处理专业**: 专注CV领域，功能丰富
3. **🛠️ 工具齐全**: 从文件操作到性能监控一应俱全
4. **⚡ 高效便捷**: 简化常用操作，提高开发效率
5. **🔧 易于使用**: API设计简洁，学习成本低
6. **📊 实用算法**: 提供IoU等常用CV算法实现

## 🔮 适用场景

- **计算机视觉项目**: 图像处理、目标检测、图像分析
- **数据科学应用**: 图像数据预处理和可视化
- **自动化脚本**: 批量图像处理和文件管理
- **研究开发**: 快速原型开发和实验验证
- **中文环境**: 需要处理中文路径和文本的应用

Cedar库是一个功能全面、易于使用的Python图像处理工具库，特别适合中文环境下的计算机视觉开发工作。 