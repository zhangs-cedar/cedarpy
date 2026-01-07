# CedarPy

<p align="left">
    <img src="https://img.shields.io/badge/python-3.8+-orange.svg">
    <img src="https://img.shields.io/badge/os-linux%2C%20windows-yellow.svg">
</p>

图像处理工具库，提供图像读写、绘图、标注等功能。

## 安装

### 从 PyPI 安装

```bash
pip install cedar
```

### 从源码安装

```bash
pip install -r requirements.txt
pip install -e .
```

或使用脚本：

```bash
sh make.sh
```

## 使用示例

```python
from cedar.image import imread, imwrite
from cedar.draw import putText, imshow

# 读取图片
img = imread('test.jpg')

# 添加文字
img = putText(img, '测试', (50, 50))

# 显示图片
imshow(img)

# 保存图片
imwrite('output.jpg', img)
```

## 格式化代码

```bash
black -l 140 cedar
```
