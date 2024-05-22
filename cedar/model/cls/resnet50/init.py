import os
import os.path as osp
import copy
import random
import platform
import chardet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import warnings
from tqdm import tqdm
from sklearn.metrics import accuracy_score
 
warnings.filterwarnings("ignore")


def get_encoding(path):
    """
    获取文件编码。
    """
    f = open(path, 'rb')
    data = f.read()
    file_encoding = chardet.detect(data).get('encoding')
    f.close()
    return file_encoding


def path_normalization(path: str):
    """
    对路径进行标准化处理，将路径中的斜杠符号（/）或反斜杠符号（\）统一成当前操作系统所支持的路径分隔符。    
    """
    win_sep = "\\"
    other_sep = "/"
    if platform.system() == "Windows":
        path = win_sep.join(path.split(other_sep))
    else:
        path = other_sep.join(path.split(win_sep))

    return path


def is_pic(names: str):
    """
    判断文件名是否为图片文件。
    """
    valid_suffix = ["JPEG", "JPG", "BMP", "PNG", "jpeg", "jpg", "bmp", "png"]
    suffix = names.split(".")[-1]
    if suffix not in valid_suffix:
        return False
    return True
