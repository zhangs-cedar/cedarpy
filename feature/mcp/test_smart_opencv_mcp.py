import os
import cv2
import numpy as np
from smart_opencv_mcp import process_image_by_description

def create_test_image(output_path: str = "test_image.jpg") -> str:
    """创建测试图像
    
    Args:
        output_path: 输出图像路径
        
    Returns:
        创建的图像路径
    """
    # 创建一个简单的测试图像
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), -1)
    cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
    
    # 添加一些噪声
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # 保存图像
    cv2.imwrite(output_path, img)
    return output_path

def test_smart_processor():
    """测试智能图像处理器"""
    # 创建测试图像
    test_image_path = create_test_image()
    
    # 测试用例1：基本处理流程
    description1 = "读取图片，进行高斯模糊(核大小:5)，然后进行边缘检测(阈值1:100,阈值2:200)"
    result1 = process_image_by_description(
        description1,
        test_image_path,
        output_dir="output/test1"
    )
    assert result1["status"] == "success", "处理失败"
    assert os.path.exists(result1["final_output"]), "输出文件未生成"
    print("测试用例1 通过")
    
    # 测试用例2：多步骤处理
    description2 = "读取图片，进行中值滤波(核大小:5)，然后进行阈值处理(阈值:127,最大值:255)，最后进行形态学操作(操作:open,核大小:3)"
    result2 = process_image_by_description(
        description2,
        test_image_path,
        output_dir="output/test2"
    )
    assert result2["status"] == "success", "处理失败"
    assert os.path.exists(result2["final_output"]), "输出文件未生成"
    print("测试用例2 通过")
    
    



if __name__ == "__main__":
    test_smart_processor()
    print("所有测试通过！") 