import logging


def logging_init(debug=True, log_path="./日志.log"):
    """日志初始化
    Args:
        debug (bool): Defaults to True.
        log_path (str): Defaults to "./日志.log".
    Notes:
        如果debug为True,则日志级别为DEBUG,否则为INFO
        log_path为日志文件路径,默认为当前目录下的日志文件
    """
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)  # 设置日志记录等级
    else:
        logger.setLevel(logging.INFO)  # 设置日志记录等级
    handler = logging.FileHandler(log_path, encoding="utf8")
    stream_handler = logging.StreamHandler()  # 往屏幕上输出
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    )
    handler.setFormatter(formatter)  # 设置写入日志格式
    stream_handler.setFormatter(formatter)  # 设置屏幕上显示格式
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
