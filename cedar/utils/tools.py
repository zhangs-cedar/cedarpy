import os
import os.path as osp
import natsort
import shutil
import hashlib
import time
import traceback
import subprocess
from typing import Any, List, Dict, Tuple, Optional, Union
from functools import wraps
from datetime import datetime
from cedar.supper import print


def create_name() -> str:
    """生成基于当前时间的唯一名称

    Returns:
        str: 格式化的时间字符串，格式为 YYYY-MM-DD_HH-MM-SS-ffffff
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


def split_filename(filename: str) -> Tuple[str, str]:
    """分解文件名为基础名称和扩展名

    Args:
        filename: 要分解的文件名

    Returns:
        Tuple[str, str]: (基础名称, 扩展名)

    Examples:
        >>> split_filename("test.txt")
        ('test', '.txt')
        >>> split_filename("test.jpg")
        ('test', '.jpg')
    """
    name = filename.rsplit(".", 1)[0]
    suffix = "." + filename.rsplit(".", 1)[1]
    return name, suffix


def rmtree_makedirs(*directories: str) -> None:
    """删除并重新创建指定的目录

    Args:
        *directories: 要处理的目录路径列表

    Example:
        >>> rmtree_makedirs("/tmp/test/a", "/tmp/test/b")
    """
    for directory in directories:
        if osp.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        print(f"[Method {rmtree_makedirs.__name__}], Create dir {directory}")


def timeit(func):
    """时间装饰器，用于测量函数执行时间

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数
    """

    @wraps(func)
    def decorated(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            print(
                f"[Method {func.__name__}], FINISH Time {round((time.time() - start), 4)} s: \n"
            )
            return result
        except Exception as e:
            error_trace = (
                str(traceback.format_exc())
                .split("func(*args, **kwargs)")[-1]
                .split("decorated")[0]
            )
            print(error_trace)
            exit(0)

    return decorated


def set_timeit_env(debug: bool = True) -> None:
    """设置或修改环境变量timeit_debug

    Args:
        debug: 是否启用调试模式，默认为True
    """
    debug_str = str(debug)
    os.environ["timeit_debug"] = debug_str
    print(f"环境变量 timeit_debug 已设置为 {debug_str}")


def run_subprocess(
    cmd: Union[str, List[str]], cwd: Optional[str] = None
) -> Tuple[subprocess.Popen, bytes, bytes]:
    """运行子进程并返回结果

    Args:
        cmd: 要执行的命令，可以是字符串或字符串列表
        cwd: 工作目录，如果为None则使用当前脚本所在目录

    Returns:
        Tuple[subprocess.Popen, bytes, bytes]: (进程对象, 标准输出, 标准错误)
    """
    if not cwd:
        script_path = os.path.abspath(__file__)
        cwd = os.path.dirname(script_path)

    # 为了安全起见，避免使用shell=True，除非绝对必要
    process = subprocess.Popen(
        cmd, cwd=cwd, shell=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )

    # 读取输出和错误，避免潜在的死锁
    stdout, stderr = process.communicate()
    return process, stdout, stderr


def get_file_md5(filename: str) -> str:
    """计算指定文件的MD5哈希值

    Args:
        filename: 文件的路径

    Returns:
        str: 文件的MD5哈希值，以十六进制字符串形式

    Raises:
        FileNotFoundError: 当文件不存在时
    """
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicate_filenames(directory: str) -> List[str]:
    """在指定目录下搜索重复的文件名

    Args:
        directory: 需要搜索的目录路径

    Returns:
        List[str]: 包含重复文件名的列表

    Raises:
        FileNotFoundError: 当目录不存在时
    """
    filename_counts: Dict[str, int] = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filename_counts[filename] = filename_counts.get(filename, 0) + 1

    return [filename for filename, count in filename_counts.items() if count > 1]


def move_file(src_path: str, dst_dir: str, filename: Optional[str] = None) -> None:
    """将文件从源路径移动到目标目录，并可选择性地重命名文件

    Args:
        src_path: 源文件的路径
        dst_dir: 目标目录的路径
        filename: 可选的文件名，如果提供则将源文件重命名为该名称

    Raises:
        FileNotFoundError: 如果源文件不存在
        PermissionError: 如果用户没有足够的权限
    """
    if filename is None:
        filename = os.path.basename(src_path)

    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, filename)

    if os.path.exists(dst_path):
        os.remove(dst_path)

    shutil.move(src_path, dst_path)


def copy_file(src_path: str, dst_dir: str, filename: Optional[str] = None) -> None:
    """将文件从源路径复制到目标目录，并可选择性地重命名文件

    Args:
        src_path: 源文件的路径
        dst_dir: 目标目录的路径
        filename: 可选的文件名，如果提供则将源文件重命名为该名称

    Raises:
        FileNotFoundError: 如果源文件不存在
        PermissionError: 如果用户没有足够的权限
    """
    if filename is None:
        filename = os.path.basename(src_path)

    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, filename)

    if os.path.exists(dst_path):
        os.remove(dst_path)

    shutil.copy(src_path, dst_path)


def get_files_list(
    input_path: str, find_suffix: Optional[List[str]] = None, sortby: str = "name"
) -> List[Dict[str, Any]]:
    """获取文件列表

    Args:
        input_path: 输入目录或文件路径
        find_suffix: 要过滤的文件扩展名列表，如果为空则不过滤
        sortby: 排序字段，默认为"name"

    Returns:
        List[Dict[str, Any]]: 文件信息列表，每个字典包含文件的各种属性

    Raises:
        ValueError: 当输入路径既不是文件也不是目录时
    """
    if find_suffix is None:
        find_suffix = []

    filepath_list: List[str] = []

    if os.path.isfile(input_path):
        filepath_list.append(input_path)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                filepath_list.append(file_path)
    else:
        raise ValueError("Input path must be a file or directory.")

    files_list: List[Dict[str, Any]] = []
    for file_path in filepath_list:
        names = osp.basename(file_path)
        root = osp.dirname(file_path)
        name, suffix = split_filename(names)

        # 如果指定了扩展名过滤且当前文件扩展名不在列表中，则跳过
        if find_suffix and suffix not in find_suffix:
            continue

        modification_time = osp.getmtime(file_path)
        file_info = {
            "name": name,
            "suffix": suffix,
            "names": names,
            "path": file_path,
            "root": root,
            "modification_time": datetime.fromtimestamp(modification_time),
        }
        files_list.append(file_info)

    return natsort.natsorted(files_list, key=lambda x: x[sortby])


def get_nested_value(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """安全地从嵌套字典中获取值

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
