import time
import logging
import cv2
import numpy as np

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    logging.warning("sklearnex not installed, using scikit-learn instead")
from sklearn.ensemble import RandomForestClassifier
from cedar.feature import multiscale_features, load_model, save_model, predict_segmenter, fit_segmenter, DataLoader
from cedar.utils import timeit


class TrainSegmentation:
    """训练ml模型"""

    def __init__(self, model_path, config):
        self.multiscale = multiscale_features(edges=False, intensity=False, texture=True, num_workers=2)
        self.model_path = model_path
        self.config = config
        self.b_height = config["b_height"]
        self.b_width = config["b_width"]
        self.train_dir = config["train_dir"]

    def _save_model(self):
        """保存模型"""
        model = {
            "clf": self.clf,
            "multiscale": self.multiscale,
            "Canny": self.config["Canny"],
            "GaussianBlur": self.config["GaussianBlur"],
            "erode_iterations": self.config["erode_iterations"],
        }
        save_model(model, self.model_path)

    def _load_data(self):
        """加载数据"""
        training_imgs, training_labels = DataLoader(self.train_dir, self.b_width)
        # 构建训练图片
        training_imgs = cv2.cvtColor(training_imgs, cv2.COLOR_BGR2GRAY)
        training_imgs = np.expand_dims(cv2.pyrDown(cv2.pyrDown(training_imgs)), axis=2)
        # 构建训练标签
        training_labels = cv2.resize(training_labels, (training_imgs.shape[1], training_imgs.shape[0]), interpolation=cv2.INTER_NEAREST)  # 线性插值

        return training_imgs, training_labels

    def train(self):
        """训练"""
        logging.info("开始训练")
        training_imgs, training_labels = self._load_data()
        logging.info("[构建训练数据] training_imgs.shape: {}".format(training_imgs.shape))
        logging.info("[构建训练数据] training_labels.shape: {}".format(training_labels.shape))

        RandomForest = RandomForestClassifier(n_estimators=20, n_jobs=-1, max_depth=20, max_samples=0.05)  # n_jobs=并行数量
        logging.info("[开始训练] RandomForest 网络: {}".format(RandomForest))
        clf = fit_segmenter(training_labels, self.multiscale(training_imgs), RandomForest)
        logging.info("[训练完成] RandomForest 网络: {}".format(RandomForest))

        self.clf = clf
        self._save_model()
        return clf


class DefaultPredictor:
    def __init__(self, model_path):
        model = load_model(model_path)
        self.model = model
        # 模型参数
        self.clf = model["clf"]
        self.multiscale = model["multiscale"]
        self.Canny = model["Canny"]
        self.GaussianBlur = model["GaussianBlur"]
        self.erode_iterations = model["erode_iterations"]

    @timeit
    def __call__(self, original_image):
        """检测单张图片,输出过滤结果

        Args:
            original_image (np.ndarray): 原始图片 (H, W, C)
        Returns:
            edges (np.ndarray): 边缘检测结果 (H, W) 0-255
            result (np.ndarray): 检测结果 (H, W) 0-255

        """
        # 图片预处理和特征提取：于训练时一致
        t1 = time.time()
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(img, (self.GaussianBlur[0], self.GaussianBlur[1]), 0), self.Canny[0], self.Canny[1])  # 高斯滤波,边缘检测
        imgp2 = np.expand_dims(cv2.pyrDown(cv2.pyrDown(img)), axis=2)
        multiscale_features = self.multiscale(imgp2)

        # 预测
        t2 = time.time()
        print("\n multiscale time: {} multiscale_features.shape: {}".format((t2 - t1), multiscale_features.shape))
        result = predict_segmenter(multiscale_features, self.clf)
        t3 = time.time()
        print("\n predict time: {}".format((t3 - t2)))

        # 后处理 1:other 2: 电池片
        _, result = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)  # 二值化:大于等于1的为255,小于1的为0 # 二值化

        result = cv2.erode(result, np.ones((3, 2), np.uint8), iterations=self.erode_iterations[0])  # 腐蚀 3, 2 主要作用于纵向
        result = cv2.dilate(result, np.ones((1, 5), np.uint8), iterations=self.erode_iterations[1])  # 膨胀 1, 5 主要作用于横向,增强横向的联通性
        result = cv2.resize(result, (edges.shape[1], edges.shape[0]), interpolation=cv2.INTER_NEAREST)  # 线性插值

        return result, edges
