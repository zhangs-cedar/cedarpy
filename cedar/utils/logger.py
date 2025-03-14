import os
import sys
import logging
from cedar.utils.tools import create_name

_logger = None


def init_logger(name=__name__, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.
    如果尚未初始化日志记录器，则通过此方法初始化日志记录器，添加一个或两个处理程序；否则将直接返回已初始化的日志记录器。
    在初始化过程中，将始终添加一个 StreamHandler。如果指定了 log_file，则还将添加一个 FileHandler。
    Args:
        name (str): 日志记录器名称，（"root"｜ 其他）
        log_file (str | None): 日志文件名。如果指定，则将向日志记录器添加一个 FileHandler。
        log_level (int): 日志记录器级别。请注意，仅影响进程0的过程，其他进程将级别设置为"Error"，因此大部分时间将保持沉默。
    Returns:
        logging.Logger: 预期的日志记录器。
    """
    global _logger
    assert _logger is None, "logger should not be initialized twice or more."
    _logger = logging.getLogger(name)

    # 修改日志格式添加文件名和行号

    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(name)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",  # 添加毫秒显示
        datefmt="%Y/%m/%d %H:%M:%S",  # 注意这里移除了末尾空格
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    if log_file is None:
        log_file = "./{}.log".format(create_name())
    log_file_folder = os.path.split(log_file)[0]
    os.makedirs(log_file_folder, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf8")
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    _logger.setLevel(log_level)
    _logger.warning("Initialize logger")


def info(fmt, *args):
    _logger.info(fmt, *args, stacklevel=2)  # 添加stacklevel参数


def debug(fmt, *args):
    _logger.debug(fmt, *args, stacklevel=2)  # 添加stacklevel参数


def warning(fmt, *args):
    _logger.warning(fmt, *args, stacklevel=2)  # 添加stacklevel参数


def error(fmt, *args):
    _logger.error(fmt, *args, stacklevel=2)  # 添加stacklevel参数
