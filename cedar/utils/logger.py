import os
import sys
import logging
from typing import Optional, Any
from cedar.utils.tools import create_name

_logger: Optional[logging.Logger] = None


def init_logger(name: str = __name__, log_file: Optional[str] = None, log_level: int = logging.INFO) -> None:
    """初始化并获取日志记录器

    如果尚未初始化日志记录器，则通过此方法初始化日志记录器，添加一个或两个处理程序；
    否则将直接返回已初始化的日志记录器。在初始化过程中，将始终添加一个 StreamHandler。
    如果指定了 log_file，则还将添加一个 FileHandler。

    Args:
        name: 日志记录器名称（"root" 或其他）
        log_file: 日志文件名，如果指定则将向日志记录器添加一个 FileHandler
        log_level: 日志记录器级别，仅影响进程0的过程，其他进程将级别设置为"Error"

    Raises:
        AssertionError: 当日志记录器被重复初始化时
    """
    global _logger
    assert _logger is None, "logger should not be initialized twice or more."
    _logger = logging.getLogger(name)

    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)

    if log_file is None:
        log_file = f"./{create_name()}.log"

    log_file_folder = os.path.split(log_file)[0]
    os.makedirs(log_file_folder, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf8")
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    _logger.setLevel(log_level)
    _logger.warning("Initialize logger")


def info(fmt: str, *args: Any) -> None:
    """记录信息级别的日志

    Args:
        fmt: 日志格式字符串
        *args: 格式化参数
    """
    if _logger:
        _logger.info(fmt, *args)


def debug(fmt: str, *args: Any) -> None:
    """记录调试级别的日志

    Args:
        fmt: 日志格式字符串
        *args: 格式化参数
    """
    if _logger:
        _logger.debug(fmt, *args)


def warning(fmt: str, *args: Any) -> None:
    """记录警告级别的日志

    Args:
        fmt: 日志格式字符串
        *args: 格式化参数
    """
    if _logger:
        _logger.warning(fmt, *args)


def error(fmt: str, *args: Any) -> None:
    """记录错误级别的日志

    Args:
        fmt: 日志格式字符串
        *args: 格式化参数
    """
    if _logger:
        _logger.error(fmt, *args)
