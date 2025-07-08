import os
import logging
import cv2
import joblib
import numpy as np
from xml.etree import ElementTree as et
from typing import Tuple, Any
from cedar.utils import split_filename


def get_labels(xml_path: str, img_width: int, img_height: int) -> np.ndarray:
    """获取标签

    Args:
        xml_path: xml 文件路径
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        np.ndarray: 标签
    """
    parsexml = et.parse(xml_path)
    root = parsexml.getroot()
    labels = np.zeros((img_height, img_width), dtype=np.uint8)
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        if name == 'other':
            labels[ymin:ymax, xmin:xmax] = 1
        else:
            labels[ymin:ymax, xmin:xmax] = 2
    return labels


def DataLoader(train_dir: str, img_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """数据加载器, 用于获取训练数据

    Args:
        train_dir: 训练集路径
        img_width: 图像宽度

    Returns:
        Tuple[np.ndarray, np.ndarray]: 图像和标签
    Notes:
        图像和标签的格式为: (图像, 标签)
        图像和标签经过两次cv2.pyrDown() 压缩, 图像的尺寸为: (img_width, img_height)
    """
    img_dir = os.path.join(train_dir, 'img')
    xml_dir = os.path.join(train_dir, 'xml')
    img_names = os.listdir(img_dir)
    logging.info(f'loading data... {len(img_names)} images')
    training_imgs = None
    training_labels = None
    for idx, img_name in enumerate(img_names):
        name, _ = split_filename(img_name)
        img_path = os.path.join(img_dir, img_name)
        xml_path = os.path.join(xml_dir, name + '.xml')
        _img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        img_height = _img.shape[0]
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        img[:, : _img.shape[1], :] = _img
        label = get_labels(xml_path, img_width, img_height)
        if idx == 0:
            training_imgs = img[None, ...]
            training_labels = label[None, ...]
        else:
            training_imgs = np.vstack((training_imgs, img[None, ...]))
            training_labels = np.vstack((training_labels, label[None, ...]))
    return training_imgs, training_labels


def load_model(filepath: str) -> Any:
    """加载预训练模型

    Args:
        filepath: 模型文件路径

    Returns:
        Any: 加载的模型对象
    """
    model = joblib.load(filepath)
    logging.info(f'[Method load_model], model loaded from {filepath}')
    return model


def save_model(model: Any, filepath: str) -> None:
    """保存模型

    Args:
        model: 需要保存的模型对象
        filepath: 保存路径
    """
    joblib.dump(model, filepath)
    logging.info(f'[Method save_model], model saved to {filepath}')


def fit_segmenter(labels: np.ndarray, features: np.ndarray, clf: Any) -> Any:
    """使用有标签的图像和分类器进行分割训练

    Args:
        labels: 图像标签
        features: 图像特征
        clf: 分类器

    Returns:
        Any: 训练好的分类器
    Notes:
        数据量不能太大, 否则会导致内存溢出
    """
    mask = labels > 0
    training_data = features[mask]
    training_labels = labels[mask].ravel()
    clf.fit(training_data, training_labels)
    return clf


def predict_segmenter(features: np.ndarray, clf: Any) -> np.ndarray:
    """使用预训练分类器进行图像分割预测

    Args:
        features: 图像特征
        clf: 分类器

    Returns:
        np.ndarray: 分割结果
    """
    sh = features.shape
    if features.ndim > 2:
        features = features.reshape((-1, sh[-1]))
    try:
        predicted_labels = clf.predict(features)
    except ValueError as err:
        if err.args and 'x must consist of vectors of length' in err.args[0]:
            raise ValueError(err.args[0] + '\n' + 'Maybe you did not use the same type of features for training the classifier.')
        else:
            raise err
    output = predicted_labels.reshape(sh[:-1])
    return output
