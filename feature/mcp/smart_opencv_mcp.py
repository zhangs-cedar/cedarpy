from typing import List, Dict, Any, Optional, Union
from mcp.server.fastmcp import FastMCP
import cv2
import numpy as np
import os
import json
from dataclasses import dataclass
from enum import Enum

class OperationType(Enum):
    """操作类型枚举"""
    READ = "read"
    FILTER = "filter"
    EDGE = "edge"
    THRESHOLD = "threshold"
    MORPHOLOGY = "morphology"
    SAVE = "save"

@dataclass
class Operation:
    """操作配置类"""
    type: OperationType
    params: Dict[str, Any]
    description: str

class SmartImageProcessor:
    """智能图像处理器"""
    
    def __init__(self):
        self.operations_map = {
            "读取图片": self._read_image,
            "高斯模糊": self._gaussian_blur,
            "中值滤波": self._median_blur,
            "双边滤波": self._bilateral_filter,
            "边缘检测": self._canny_edge,
            "阈值处理": self._threshold,
            "形态学操作": self._morphology,
            "保存图片": self._save_image
        }
        
        self.param_patterns = {
            "高斯模糊": r"核大小[：:]\s*(\d+)",
            "中值滤波": r"核大小[：:]\s*(\d+)",
            "双边滤波": r"直径[：:]\s*(\d+).*?颜色空间[：:]\s*(\d+).*?空间[：:]\s*(\d+)",
            "边缘检测": r"阈值1[：:]\s*(\d+).*?阈值2[：:]\s*(\d+)",
            "阈值处理": r"阈值[：:]\s*(\d+).*?最大值[：:]\s*(\d+)",
            "形态学操作": r"操作[：:]\s*(\w+).*?核大小[：:]\s*(\d+)"
        }
    
    def _read_image(self, img: Optional[np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """读取图片"""
        path = params.get("path")
        if not path:
            raise ValueError("未指定图片路径")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图片: {path}")
        return img
    
    def _gaussian_blur(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """高斯模糊"""
        ksize = params.get("ksize", 5)
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    def _median_blur(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """中值滤波"""
        ksize = params.get("ksize", 5)
        return cv2.medianBlur(img, ksize)
    
    def _bilateral_filter(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """双边滤波"""
        d = params.get("d", 9)
        sigma_color = params.get("sigma_color", 75)
        sigma_space = params.get("sigma_space", 75)
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    def _canny_edge(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Canny边缘检测"""
        threshold1 = params.get("threshold1", 100)
        threshold2 = params.get("threshold2", 200)
        return cv2.Canny(img, threshold1, threshold2)
    
    def _threshold(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """阈值处理"""
        thresh = params.get("thresh", 127)
        maxval = params.get("maxval", 255)
        _, result = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
        return result
    
    def _morphology(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """形态学操作"""
        op_type = params.get("op_type", "open")
        ksize = params.get("ksize", 5)
        kernel = np.ones((ksize, ksize), np.uint8)
        
        if op_type == "open":
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif op_type == "close":
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        elif op_type == "dilate":
            return cv2.dilate(img, kernel, iterations=1)
        elif op_type == "erode":
            return cv2.erode(img, kernel, iterations=1)
        else:
            raise ValueError(f"不支持的形态学操作类型: {op_type}")
    
    def _save_image(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """保存图片"""
        path = params.get("path")
        if not path:
            raise ValueError("未指定保存路径")
        cv2.imwrite(path, img)
        return img

mcp = FastMCP("Smart-OpenCV-Service")
processor = SmartImageProcessor()

@mcp.tool()
def process_image_by_description(
    description: str,
    input_path: str,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """根据自然语言描述处理图像
    
    Args:
        description: 处理流程的自然语言描述
        input_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        处理结果信息
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析处理流程
    operations = parse_operations(description)
    
    # 执行处理流程
    img = None
    results = []
    
    for i, op in enumerate(operations):
        # 更新操作参数
        if op.type == OperationType.READ:
            op.params["path"] = input_path
        elif op.type == OperationType.SAVE:
            base_name = os.path.basename(input_path)
            name, ext = os.path.splitext(base_name)
            op.params["path"] = os.path.join(output_dir, f"{name}_step{i}{ext}")
        
        # 执行操作
        img = processor.operations_map[op.description](img, op.params)
        
        # 记录结果
        results.append({
            "step": i + 1,
            "operation": op.description,
            "params": op.params,
            "output_path": op.params.get("path") if op.type == OperationType.SAVE else None
        })
    
    return {
        "status": "success",
        "operations": results,
        "final_output": results[-1]["output_path"] if results else None
    }

def parse_operations(description: str) -> List[Operation]:
    """解析自然语言描述为操作列表
    
    Args:
        description: 处理流程的自然语言描述
        
    Returns:
        操作列表
    """
    # 这里可以接入LLM来解析自然语言
    # 目前使用简单的规则匹配
    operations = []
    
    # 示例：解析"读取图片，进行高斯模糊(核大小:5)，然后进行边缘检测(阈值1:100,阈值2:200)"
    if "读取图片" in description:
        operations.append(Operation(OperationType.READ, {}, "读取图片"))
    
    if "高斯模糊" in description:
        ksize = 5  # 默认值
        if "核大小" in description:
            # 这里可以添加更复杂的参数提取逻辑
            pass
        operations.append(Operation(OperationType.FILTER, {"ksize": ksize}, "高斯模糊"))
    
    if "边缘检测" in description:
        threshold1 = 100  # 默认值
        threshold2 = 200  # 默认值
        if "阈值1" in description and "阈值2" in description:
            # 这里可以添加更复杂的参数提取逻辑
            pass
        operations.append(Operation(
            OperationType.EDGE,
            {"threshold1": threshold1, "threshold2": threshold2},
            "边缘检测"
        ))
    
    # 添加保存操作
    operations.append(Operation(OperationType.SAVE, {}, "保存图片"))
    
    return operations

if __name__ == "__main__":
    mcp.run() 