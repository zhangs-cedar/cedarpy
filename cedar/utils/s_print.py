import os
import datetime
from typing import Any, List, Optional

# 定义全局日志文件路径
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
LOG_FILE = os.path.join(script_directory, 's_print.log')


# 颜色代码定义 (参考Python解释器颜色方案)
class Colors:
    """颜色代码定义类"""

    RESET = '\033[0m'
    GREEN = '\033[32m'  # 字符串
    BRIGHT_BLUE = '\033[94m'  # 数字
    BRIGHT_YELLOW = '\033[93m'  # 布尔值
    BRIGHT_MAGENTA = '\033[95m'  # 列表/元组
    BRIGHT_CYAN = '\033[96m'  # 字典/集合
    BRIGHT_BLACK = '\033[90m'  # None
    BRIGHT_GREEN = '\033[92m'  # 函数/类型
    WHITE = '\033[37m'  # 默认


# 类型颜色映射
TYPE_COLORS = {
    'str': Colors.GREEN,
    'int': Colors.BRIGHT_BLUE,
    'float': Colors.BRIGHT_BLUE,
    'bool': Colors.BRIGHT_YELLOW,
    'list': Colors.BRIGHT_MAGENTA,
    'tuple': Colors.BRIGHT_MAGENTA,
    'dict': Colors.BRIGHT_CYAN,
    'set': Colors.BRIGHT_CYAN,
    'NoneType': Colors.BRIGHT_BLACK,
    'function': Colors.BRIGHT_GREEN,
    'type': Colors.BRIGHT_GREEN,
    'module': Colors.BRIGHT_GREEN,
}


def get_type_color(arg_type: str) -> str:
    """获取类型对应的颜色代码"""
    # 处理特殊类型名称
    if arg_type.startswith('class'):
        arg_type = 'type'
    elif arg_type.startswith('function'):
        arg_type = 'function'
    elif arg_type.startswith('module'):
        arg_type = 'module'

    return TYPE_COLORS.get(arg_type, Colors.WHITE)


def format_arg_with_color(arg: Any, arg_type: str) -> str:
    """格式化参数并添加颜色"""
    color_code = get_type_color(arg_type)

    if isinstance(arg, str):
        return f'{color_code}{arg}{Colors.RESET}    '
    else:
        return f'{color_code}({arg_type}) {arg}{Colors.RESET}    '


# 备份原始的 print 函数
original_print = print


def print(*args: Any, sep: str = ' ', end: str = '\n', file: Optional[str] = None) -> str:
    """自定义打印函数，支持颜色输出和日志记录

    Args:
        *args: 要打印的参数
        sep: 分隔符，默认为空格
        end: 结束符，默认为换行
        file: 输出文件路径，默认为环境变量LOG_PATH或默认日志文件

    Returns:
        格式化后的输出字符串
    """
    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f'[{current_time}]   '

    # 格式化参数并添加颜色
    output_with_type: List[str] = []
    for arg in args:
        arg_type = type(arg).__name__
        formatted_arg = format_arg_with_color(arg, arg_type)
        output_with_type.append(formatted_arg)

    output = prefix + sep.join(output_with_type)

    # 控制台输出（带颜色）
    original_print(output, end=end)

    # 日志文件输出（不带颜色）
    log_output = prefix
    for arg in args:
        arg_type = type(arg).__name__
        if isinstance(arg, str):
            log_output += f'{arg}    '
        else:
            log_output += f'({arg_type}) {arg}    '

    # 选择日志文件路径
    log_file_path = os.environ.get('LOG_PATH', LOG_FILE)
    if file is None:
        file = log_file_path

    # 写入日志文件（不带颜色）
    with open(file, 'a', encoding='utf-8') as log_file:
        log_file.write(log_output + '\n')

    return output


if __name__ == '__main__':
    # 测试不同类型的颜色输出
    print('Hello World', 1.1, [1123], {'key': 'value'}, True, None)
