import logging
import os
import os.path as osp
import shutil
import time
import traceback
from functools import wraps


def split_filename(filename: str) -> tuple:
    """分解名字为 name 和 suffix

    Examples:
        >>> split_filename("test.txt")
        ('test', '.txt')
        >>> split_filename("test.jpg")
        ('test', '.jpg')

    Returns:
        name, suffix
    """
    name = filename.rsplit(".", 1)[0]
    suffix = "." + filename.rsplit(".", 1)[1]

    return name, suffix


def rmtree_makedirs(*args):
    """rmtree_makedirs
    Args:
        dir (str): 文件夹路径
    Example:
        >>> rmtree_makedirs("/tmp/test/a", "/tmp/test/b")
    """
    for index, dir in enumerate(args):
        if osp.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        print("[Method {}], Create dir {}".format(rmtree_makedirs.__name__, dir))


def timeit(func):
    """时间修饰器"""

    @wraps(func)
    def decorated(*args, **kwargs):
        start = time.time()
        try:
            res = func(*args, **kwargs)
            logging.info("[Method {}], FINISH Time {} s: \n".format(func.__name__, round((time.time() - start), 4)))
            return res
        except:
            logging.error(str(traceback.format_exc()).split("func(*args, **kwargs)")[-1].split("decorated")[0])
            exit(0)

    return decorated
