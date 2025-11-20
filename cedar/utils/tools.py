import hashlib
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
import traceback
import natsort
from datetime import datetime
from functools import wraps
from typing import Any

from cedar.utils.s_print import print


def init_cfg(config_file_path, file=None):
    """初始化配置"""
    file = file or __file__
    config_file_path = config_file_path or os.environ.get('SCRIPT_CONFIG_FILE') or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not config_file_path:
        raise ValueError('未提供配置文件路径')

    base_dir = os.environ.get('CEDAR_BASE_DIR', './')
    script_name = osp.basename(osp.dirname(file))
    log_dir = osp.join(base_dir, 'log', script_name)
    os.makedirs(log_dir, exist_ok=True)
    os.environ['LOG_PATH'] = osp.join(log_dir, create_name() + '.log')

    with open(config_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_name(format_type='standard'):
    """创建时间戳名字"""
    now = datetime.now()
    formats = {
        'full': '%Y-%m-%d_%H-%M-%S-%f',
        'standard': '%Y-%m-%d_%H-%M-%S',
        'date_only': '%Y-%m-%d',
        'time_only': '%H-%M-%S',
        'compact': '%Y%m%d%H%M%S%f',
    }
    if format_type not in formats:
        raise ValueError(f'不支持的格式: {format_type}')
    return now.strftime(formats[format_type])


def split_filename(filename: str) -> tuple:
    """分解文件名"""
    parts = filename.rsplit('.', 1)
    return parts[0], '.' + parts[1] if len(parts) > 1 else ''


def rmtree_makedirs(*dirs):
    """删除并重建目录"""
    for dir_path in dirs:
        if osp.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        print(f'[{rmtree_makedirs.__name__}] 创建目录 {dir_path}')


def timeit(func):
    """计时装饰器"""

    @wraps(func)
    def decorated(*args, **kwargs):
        start = time.time()
        try:
            res = func(*args, **kwargs)
            print('[Method {}], FINISH Time {} s: \n'.format(func.__name__, round((time.time() - start), 4)))
            return res
        except Exception as e:
            print(str(traceback.format_exc()).split('func(*args, **kwargs)')[-1].split('decorated')[0])
            raise

    return decorated


def try_except(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(str(traceback.format_exc()).split('func(*args, **kwargs)')[-1].split('decorated')[0])
            raise

    return decorated


def run_subprocess(cmd, cwd=None):
    """执行子进程"""
    cwd = cwd or os.path.dirname(os.path.abspath(__file__))
    process = subprocess.Popen(cmd, cwd=cwd, shell=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process, stdout, stderr


def get_file_md5(filename):
    """计算文件MD5"""
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicate_filenames(directory):
    """查找重复文件名"""
    filename_counts = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filename_counts[filename] = filename_counts.get(filename, 0) + 1
    return [f for f, count in filename_counts.items() if count > 1]


def move_file(src_path, dst_dir, filename=None):
    """移动文件"""
    filename = filename or os.path.basename(src_path)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, filename)
    if os.path.exists(dst_path):
        os.remove(dst_path)
    shutil.move(src_path, dst_path)


def copy_file(src_path, dst_dir, filename=None):
    """硬链接优先,失败fallback到copy"""
    filename = filename or os.path.basename(src_path)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, filename)

    if os.path.exists(dst_path):
        os.remove(dst_path)
    try:
        os.link(src_path, dst_path)  # 硬链接,零拷贝
    except (OSError, NotImplementedError):
        shutil.copy2(src_path, dst_path)  # 跨分区或不支持时用copy


def get_files_list(input_path, find_suffix=None, sortby='name'):
    """获取文件列表"""
    find_suffix = find_suffix or []
    filepath_list = []

    if os.path.isfile(input_path):
        filepath_list.append(input_path)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            filepath_list.extend(os.path.join(root, f) for f in files)
    else:
        raise ValueError('路径必须是文件或目录')

    files_list = []
    for file_path in filepath_list:
        names = osp.basename(file_path)
        name, suffix = split_filename(names)
        if find_suffix and suffix not in find_suffix:
            continue

        files_list.append(
            {
                'name': name,
                'suffix': suffix,
                'names': names,
                'path': file_path,
                'root': osp.dirname(file_path),
                'modification_time': datetime.fromtimestamp(osp.getmtime(file_path)),
                'creation_time': datetime.fromtimestamp(osp.getctime(file_path)),
            }
        )

    return natsort.natsorted(files_list, key=lambda x: x[sortby])


def get_nested_value(d: dict, *keys: str, default: Any = None) -> Any:
    """从嵌套字典获取值"""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d
