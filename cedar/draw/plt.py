import cv2
import matplotlib.pyplot as plt


def imshow(img):
    """显示BGR图像"""
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
