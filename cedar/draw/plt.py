import cv2
import matplotlib.pyplot as plt
import numpy as np


def imshow(img: np.ndarray) -> None:
    """以RGB方式展示BGR图像

    Args:
        img: 输入BGR格式图像
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
