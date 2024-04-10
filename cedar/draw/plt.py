import cv2
import matplotlib.pyplot as plt


def imshow(img):
    """
    img : BGR  to RGB  展示
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(img)
