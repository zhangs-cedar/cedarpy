import logging
import os
from xml.etree import ElementTree as et

import cv2
import joblib
import numpy as np
from cedar.utils import split_filename


def get_labels(xml_path: str, IMG_WIDTH: int, IMG_HEIGHT: int) -> np.ndarray:
    """获取标签
    Args:
        xml_path (str): xml 文件路径
        IMG_WIDTH (int): 图像宽度
        IMG_HEIGHT (int): 图像高度
    Returns:
        np.ndarray: 标签
    """
    parsexml = et.parse(xml_path)
    root = parsexml.getroot()
    labels = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        xmax = int(bndbox.find("xmax").text)
        ymin = int(bndbox.find("ymin").text)
        ymax = int(bndbox.find("ymax").text)
        if name == "other":
            labels[ymin:ymax, xmin:xmax] = 1
        else:
            labels[ymin:ymax, xmin:xmax] = 2

    return labels


def DataLoader(train_dir: str, IMG_WIDTH) -> object:
    """数据加载器, 用于获取训练数据

    Args:
        train_dir (str): 训练集路径
        IMG_WIDTH (int): 图像宽度
        IMG_HEIGHT (int): 图像高度

    Returns:
        object: 图像和标签
    Notes:
        图像和标签的格式为: (图像, 标签)
        图像和标签经过两次cv2.pyrDown() 压缩, 图像的尺寸为: (IMG_WIDTH, IMG_HEIGHT)
    """

    logging.info("loading data...".format(len(os.listdir(os.path.join(train_dir, "img")))))
    for idx, img_name in enumerate(os.listdir(os.path.join(train_dir, "img"))):
        name, suffix = split_filename(img_name)
        # 兼容中文路径
        img_path = os.path.join(train_dir, "img", img_name)
        xml_path = os.path.join(train_dir, "xml", name + ".xml")

        _img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)  # 读取图片、兼容中文路径
        IMG_HEIGHT = _img.shape[0]
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        img[:, : _img.shape[1], :] = _img

        if idx == 0:
            training_imgs = np.zeros((1, IMG_WIDTH, 3), dtype=np.uint8)
            training_labels = np.zeros((1, IMG_WIDTH), dtype=np.uint8)

        training_imgs = np.vstack((training_imgs, img))
        training_labels = np.vstack((training_labels, get_labels(xml_path, IMG_WIDTH, IMG_HEIGHT)))
    return training_imgs, training_labels


def load_model(filepath: str) -> object:
    """Load a pretrained classifier.
    Args:
        filepath (str): path to the pretrained classifier
    Returns:
        object: pretrained classifier
    Notes:
        The classifier is a joblib.load()d object.
    """
    model = joblib.load(filepath)
    logging.info("[Method {}], model loaded from {}".format("load_model", filepath))
    return model


def save_model(model: object, filepath: str) -> object:
    """Save a pretrained classifier.
    Args:
        model (object): pretrained classifier
        filepath (str): path to the pretrained classifier
    Notes:
        The classifier is a joblib.dump() object.
    """
    joblib.dump(model, filepath)
    logging.info("[Method {}], model saved to {}".format("save_model", filepath))
    return None


def fit_segmenter(labels: np.ndarray, features: np.ndarray, clf: object) -> object:
    """Segmentation using labeled parts of the image and a classifier.
    Args:
        labels (np.ndarray): labeled parts of the image
        features (np.ndarray): features of the image
        clf (object): classifier
    Returns:
        object: segmented image
    Notes:
        数据量不能太大, 否则会导致内存溢出
    """
    mask = labels > 0
    training_data = features[mask]
    training_labels = labels[mask].ravel()
    clf.fit(training_data, training_labels)
    return clf


def predict_segmenter(features: object, clf: object) -> object:
    """Segmentation of images using a pretrained classifier.
    Args:
        features (object): features of the image
        clf (object): classifier
    Returns:
        object: segmented image
    """
    sh = features.shape
    if features.ndim > 2:
        features = features.reshape((-1, sh[-1]))
    try:
        predicted_labels = clf.predict(features)
    except ValueError as err:
        if err.args and "x must consist of vectors of length" in err.args[0]:
            raise ValueError(err.args[0] + "\n" + "Maybe you did not use the same type of features for training the classifier.")
        else:
            raise err
    output = predicted_labels.reshape(sh[:-1])
    return output
