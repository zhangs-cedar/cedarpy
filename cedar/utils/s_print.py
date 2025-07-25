import os
import datetime

# 定义全局日志文件路径
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
LOG_FILE = os.path.join(script_directory, 's_print.log')

# 备份原始的 print 函数
original_print = print


# 定义自定义的 print 函数
def print(*args, sep=' ', end='\n', file=None):
    # 在输出前添加一个前缀
    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f'[{current_time}]   '  # 添加时间戳到日志内容
    output_with_type = []
    for arg in args:
        arg_type = type(arg).__name__
        # 如果arg是字符串，则添加颜色,不需要加（str）提示
        if isinstance(arg, str):
            output_with_type.append(f'{arg}    ')
        else:
            output_with_type.append(f'({arg_type}) {arg}    ')
    output = sep.join(output_with_type)
    output = prefix + output
    original_print(output, end=end)

    # 选择日志文件路径
    log_file_path = os.environ.get('LOG_PATH', LOG_FILE)
    if file is None:
        file = log_file_path
    with open(file, 'a', encoding='utf-8') as log_file:  # 使用追加模式
        log_file.write(output + '\n')  # 写入内容并换行
    return output


if __name__ == '__main__':
    # 使用自定义的 print 函数
    print('Hello World', 1.1, [1123])
