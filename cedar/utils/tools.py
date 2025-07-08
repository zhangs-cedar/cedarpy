import os
import os.path as osp
import natsort
import shutil
import hashlib
import time
import traceback
import subprocess
from functools import wraps
from datetime import datetime
from typing import Optional, List, Tuple, Union, Any, Dict

def create_name():
    strtime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')  # 格式化日期时间，年月日时分秒毫秒
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
    name = filename.rsplit('.', 1)[0]
    suffix = '.' + filename.rsplit('.', 1)[1]

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
        print('[Method {}], Create dir {}'.format(rmtree_makedirs.__name__, dir))


def timeit(func):
    """时间修饰器，通过环境变量 timeit_debug 控制是否打印时间信息"""

    @wraps(func)
    def decorated(*args, **kwargs):
        start = time.time()
        try:
            res = func(*args, **kwargs)
            if os.getenv('timeit_debug', 'false').lower() == 'true':
                print('[Method {}], FINISH Time {} s: \n'.format(func.__name__, round((time.time() - start), 4)))
            return res
        except Exception as e:
            if os.getenv('timeit_debug', 'false').lower() == 'true':
                print(str(traceback.format_exc()).split('func(*args, **kwargs)')[-1].split('decorated')[0])
            exit(0)

    return decorated


def set_timeit_env(debug=True):
    """
    设置或修改环境变量timeit_debug
    """
    # 将 debug 参数转换为字符串
    debug_str = str(debug)
    # 直接设置环境变量 timeit_debug 的值
    os.environ['timeit_debug'] = debug_str
    print(f'环境变量 timeit_debug 已设置为 {debug_str}')


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


def get_file_md5(filename):
    """
    计算指定文件的MD5哈希值。

    参数:
    - filename: 文件的路径。

    返回:
    - 文件的MD5哈希值，以十六进制字符串形式。
    """
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:  # 以二进制形式读取文件
        for chunk in iter(lambda: f.read(4096), b''):  # 按块读取文件
            hash_md5.update(chunk)  # 更新MD5哈希值
    return hash_md5.hexdigest()  # 返回十六进制形式的哈希值


def find_duplicate_filenames(directory):
    """
    在指定目录下搜索重复的文件名，并返回这些文件名列表。

    Args:
        directory (str): 需要搜索的目录路径。

    Returns:
        list: 包含重复文件名的列表，列表中的元素是字符串类型。

    """
    filename_counts = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename in filename_counts:
                filename_counts[filename] += 1
            else:
                filename_counts[filename] = 1
    duplicates = [f for f, count in filename_counts.items() if count > 1]
    return duplicates


def move_file(src_path, dst_dir, filename=None):
    """
    将文件从源路径移动到目标目录，并可选择性地重命名文件。

    Args:
        src_path (str): 源文件的路径。
        dst_dir (str): 目标目录的路径。
        filename (str, optional): 可选的文件名，如果提供，则将源文件重命名为该名称。
            默认为None，表示使用源文件的原始文件名。

    Returns:
        None

    Raises:
        FileNotFoundError: 如果源文件不存在，则引发此异常。
        PermissionError: 如果用户没有足够的权限来移动文件或创建目录，则可能引发此异常。
        其他可能的异常: 调用os和shutil模块时可能引发的其他异常。

    """
    # 如果没有提供文件名，则使用源文件的文件名
    if filename is None:
        filename = os.path.basename(src_path)
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)
    # 目标文件的完整路径
    dst_path = os.path.join(dst_dir, filename)
    # 如果目标目录中存在同名文件，则删除它
    if os.path.exists(dst_path):
        os.remove(dst_path)
    # 移动文件
    shutil.move(src_path, dst_path)


def copy_file(src_path, dst_dir, filename=None):
    """
    将文件从源路径移动到目标目录，并可选择性地重命名文件。

    Args:
        src_path (str): 源文件的路径。
        dst_dir (str): 目标目录的路径。
        filename (str, optional): 可选的文件名，如果提供，则将源文件重命名为该名称。
            默认为None，表示使用源文件的原始文件名。

    Returns:
        None

    Raises:
        FileNotFoundError: 如果源文件不存在，则引发此异常。
        PermissionError: 如果用户没有足够的权限来移动文件或创建目录，则可能引发此异常。
        其他可能的异常: 调用os和shutil模块时可能引发的其他异常。

    """
    # 如果没有提供文件名，则使用源文件的文件名
    if filename is None:
        filename = os.path.basename(src_path)
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)
    # 目标文件的完整路径
    dst_path = os.path.join(dst_dir, filename)
    # 如果目标目录中存在同名文件，则删除它
    if os.path.exists(dst_path):
        os.remove(dst_path)
    # 移动文件
    shutil.copy(src_path, dst_path)


def get_files_list(input_path, find_suffix=[], sortby='name'):
    """获取文件列表
    Args:
        input_path: 输入目录 | 文件路径
        return: 文件路径列表
    """
    filepath_list = []

    if os.path.isfile(input_path):  # 如果是文件
        filepath_list.append(input_path)
    elif os.path.isdir(input_path):  # 如果是目录
        for root, dirs, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                filepath_list.append(file_path)
    else:
        raise ValueError('Input path must be a file or directory.')
    # 按照文件名排序
    files_list = []
    for file_path in filepath_list:
        names = osp.basename(file_path)
        root = osp.dirname(file_path)
        name, suffix = split_filename(names)
        if len(find_suffix) != 0 and suffix not in find_suffix:  # 如果find_suffix为空，则不跳过，else找不到指定的后缀，则跳过,
            continue
        modification_time = osp.getmtime(file_path)  # 获取文件的修改时间
        # 获取文件的创建时间
        creation_time = osp.getctime(file_path)
        file = {}
        file['name'] = name
        file['suffix'] = suffix
        file['names'] = names
        file['path'] = file_path
        file['root'] = root
        file['modification_time'] = datetime.fromtimestamp(modification_time)
        file['creation_time'] = datetime.fromtimestamp(creation_time)
        files_list.append(file)
    files_list = natsort.natsorted(files_list, key=lambda x: x[sortby])
    return files_list


def get_nested_value(d: dict, *keys: str, default: Any = None) -> Any:
    """安全地从嵌套字典中获取值。

    递归地遍历嵌套字典，按照给定的键序列获取值。如果任何一个键不存在，
    则返回默认值。

    Args:
        d: 要查询的字典
        *keys: 键的序列，按照从外到内的顺序
        default: 当键不存在时返回的默认值，默认为None

    Returns:
        Any: 找到的值或默认值

    Examples:
        >>> d = {'a': {'b': {'c': 1}}}
        >>> get_nested_value(d, 'a', 'b', 'c')
        1
        >>> get_nested_value(d, 'a', 'b', 'd', default=0)
        0
    """
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d