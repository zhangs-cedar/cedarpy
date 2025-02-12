import os
import datetime
from pync import Notifier


# 定义全局日志文件路径
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
LOG_FILE = os.path.join(script_directory, "s_print.log")


def s_print(title):
    """
    A simple print function that prints to console and writes to log file with timestamp.
    """
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{current_time}] {title}"  # 添加时间戳到日志内容
    # 打印到控制台
    print(log_message)
    # 写入到日志文件
    with open(LOG_FILE, "a") as log_file:  # 使用追加模式
        log_file.write(log_message + "\n")  # 写入内容并换行
    Notifier.notify(title)


if __name__ == "__main__":
    s_print("Hello World")
