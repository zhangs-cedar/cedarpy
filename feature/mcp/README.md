# OpenCV MCP 服务

这是一个基于FastMCP的OpenCV图像处理服务，提供了多种图像处理功能。

## 功能特性

### 1. 基础图像处理
- 边缘检测
- 高斯模糊
- 阈值处理
- 目标检测（基于YOLOv3）

### 2. 智能图像处理
支持通过自然语言描述来执行图像处理流程，例如：
```python
# 示例1：基本处理流程
description = "读取图片，进行高斯模糊(核大小:5)，然后进行边缘检测(阈值1:100,阈值2:200)"
result = process_image_by_description(description, "input.jpg")

# 示例2：多步骤处理
description = "读取图片，进行中值滤波(核大小:5)，然后进行阈值处理(阈值:127,最大值:255)，最后进行形态学操作(操作:open,核大小:3)"
result = process_image_by_description(description, "input.jpg")
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 启动服务

```bash
# 基础服务
python opencv_mcp.py

# 智能处理服务
python smart_opencv_mcp.py
```

### 2. 基础图像处理示例

```python
from opencv_mcp import process_image

# 边缘检测
result_path = process_image(
    "input.jpg",
    operations=["edge_detection"],
    output_dir="output"
)

# 多种操作组合
result_path = process_image(
    "input.jpg",
    operations=["blur", "threshold"],
    output_dir="output"
)
```

### 3. 智能图像处理示例

```python
from smart_opencv_mcp import process_image_by_description

# 使用自然语言描述处理流程
description = "读取图片，进行高斯模糊(核大小:5)，然后进行边缘检测(阈值1:100,阈值2:200)"
result = process_image_by_description(
    description,
    "input.jpg",
    output_dir="output"
)

# 查看处理结果
print(f"处理状态: {result['status']}")
print(f"处理步骤: {result['operations']}")
print(f"最终输出: {result['final_output']}")
```

## 支持的操作类型

1. 基础操作
   - 读取图片
   - 保存图片

2. 滤波操作
   - 高斯模糊
   - 中值滤波
   - 双边滤波

3. 特征提取
   - 边缘检测
   - 阈值处理

4. 形态学操作
   - 开运算
   - 闭运算
   - 膨胀
   - 腐蚀

## 测试

运行测试脚本：

```bash
# 基础服务测试
python test_opencv_mcp.py

# 智能处理服务测试
python test_smart_opencv_mcp.py
```

## 在Cursor中使用

### 1. 快速启动
```bash
# 方法1: 使用启动脚本
python start_mcp.py

# 方法2: 直接启动服务
python smart_opencv_mcp.py
```

### 2. 在Cursor中配置MCP
1. 将 `mcp_config.json` 添加到你的Cursor配置中
2. 重启Cursor，服务将自动加载
3. 现在可以通过Cursor的AI助手直接调用图像处理功能

### 3. 使用示例
```bash
# 运行完整的使用示例
python example_usage.py
```

### 4. 在聊天中使用
在Cursor的AI聊天中，你可以直接请求图像处理任务：
- "帮我对这张图片进行边缘检测"
- "使用高斯模糊处理图像，然后进行阈值化"
- "执行完整的图像预处理流程：降噪 → 模糊 → 边缘检测"

## 注意事项

1. 目标检测功能需要下载YOLOv3的配置文件和权重文件：
   - yolov3.cfg
   - yolov3.weights

2. 确保有足够的磁盘空间用于存储处理后的图像。

3. 建议在处理大图像时适当调整参数以获得更好的性能。

4. 智能处理服务目前支持中文描述，后续会添加英文支持。

5. 参数提取目前使用简单的规则匹配，后续会接入LLM实现更智能的解析。

6. 在Cursor中使用时，确保MCP服务正在后台运行。 