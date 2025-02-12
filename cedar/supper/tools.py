import os
import datetime

try:
    from pync import Notifier
except:
    pass

# 定义全局日志文件路径
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
LOG_FILE = os.path.join(script_directory, "s_print.log")

# 备份原始的 print 函数
original_print = print


# 定义自定义的 print 函数
def print(*args, sep=" ", end="\n", file=None):
    # 在输出前添加一个前缀
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[Print] [{current_time}]   "  # 添加时间戳到日志内容
    output_with_type = []
    for arg in args:
        arg_type = type(arg).__name__
        output_with_type.append(f"({arg_type}) {arg}    ")
    output = sep.join(output_with_type)
    original_print(prefix + output, end=end)
    # 将输出写入到日志文件
    if file is None:
        file = LOG_FILE
    with open(file, "a", encoding="utf-8") as log_file:  # 使用追加模式
        log_file.write(output + "\n")  # 写入内容并换行
    try:
        _ = sep.join(str(arg) for arg in args)
        Notifier.notify(_)
    except:
        pass
    return output


if __name__ == "__main__":
    # 使用自定义的 print 函数
    print("Hello World", 1.1, [1123])
