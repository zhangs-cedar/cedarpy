import os
from datetime import datetime
from typing import Any, Optional

LOG_FILE = os.path.join(os.path.dirname(__file__), 's_print.log')

# 颜色映射
COLORS = {
    'str': '\033[32m',
    'int': '\033[94m',
    'float': '\033[94m',
    'bool': '\033[93m',
    'list': '\033[95m',
    'tuple': '\033[95m',
    'dict': '\033[96m',
    'set': '\033[96m',
    'NoneType': '\033[90m',
    'function': '\033[92m',
    'type': '\033[92m',
    'module': '\033[92m',
    'reset': '\033[0m',
    'default': '\033[37m',
}


def format_arg(arg: Any) -> tuple:
    """格式化参数"""
    arg_type = type(arg).__name__

    # 处理特殊类型
    if arg_type.startswith(('class', 'function', 'module')):
        arg_type = arg_type.split()[0] if ' ' in arg_type else arg_type

    color = COLORS.get(arg_type, COLORS['default'])
    reset = COLORS['reset']

    if isinstance(arg, str):
        colored = f'{color}{arg}{reset}    '
        plain = f'{arg}    '
    else:
        colored = f'{color}({arg_type}) {arg}{reset}    '
        plain = f'({arg_type}) {arg}    '

    return colored, plain


original_print = print


def print(*args: Any, sep: str = ' ', end: str = '\n', file: Optional[str] = None) -> str:
    """带颜色和日志的打印函数"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f'[{timestamp}]   '

    # 格式化所有参数
    colored_parts = []
    plain_parts = []

    for arg in args:
        colored, plain = format_arg(arg)
        colored_parts.append(colored)
        plain_parts.append(plain)

    # 控制台输出（带颜色）
    colored_output = prefix + sep.join(colored_parts)
    original_print(colored_output, end=end)

    # 日志输出（无颜色）
    plain_output = prefix + sep.join(plain_parts)
    log_path = file or os.environ.get('LOG_PATH', LOG_FILE)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(plain_output + '\n')

    return colored_output


if __name__ == '__main__':
    print('Hello World', 1.1, [1123], {'key': 'value'}, True, None)
