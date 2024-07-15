import os
import os.path as osp
import shutil
import time
import datetime
import traceback
import subprocess
from functools import wraps


def create_name():
    strtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")  # 格式化日期时间，年月日时分秒毫秒
    return strtime


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
            print("[Method {}], FINISH Time {} s: \n".format(func.__name__, round((time.time() - start), 4)))
            return res
        except:
            print(str(traceback.format_exc()).split("func(*args, **kwargs)")[-1].split("decorated")[0])
            exit(0)

    return decorated


def run_subprocess(cmd, cwd=None):
    if not cwd:
        # 获取当前脚本文件的路径
        script_path = os.path.abspath(__file__)
        # 获取当前脚本文件所在的目录
        cwd = os.path.dirname(script_path)

    # 为了安全起见，避免使用shell=True，除非绝对必要。
    # 如果确实需要使用shell特性，确保cmd的内容是安全的。
    process = subprocess.Popen(cmd, cwd=cwd, shell=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    # 读取输出和错误，避免潜在的死锁
    stdout, stderr = process.communicate()
    return process, stdout, stderr
