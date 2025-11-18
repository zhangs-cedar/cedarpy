import logging
import os
import sys
from typing import Any

from cedar.utils.tools import create_name

_logger = None


def init_logger(name=__name__, log_file=None, log_level=logging.INFO):
    """初始化日志器"""
    global _logger
    if _logger:
        return

    _logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    # 控制台输出
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    _logger.addHandler(console)

    # 文件输出
    log_file = log_file or f'./{create_name()}.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding='utf8')
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)

    _logger.setLevel(log_level)
    _logger.warning('日志器初始化完成')


def _log(level, fmt, *args):
    """统一日志记录"""
    if _logger:
        getattr(_logger, level)(fmt, *args)


def info(fmt, *args):
    _log('info', fmt, *args)


def debug(fmt, *args):
    _log('debug', fmt, *args)


def warning(fmt, *args):
    _log('warning', fmt, *args)


def error(fmt, *args):
    _log('error', fmt, *args)
