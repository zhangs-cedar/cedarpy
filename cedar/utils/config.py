import os
import json5 as json
import yaml
from cedar.utils.s_print import print

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config(config_file=None):
    """加载配置文件"""
    config_file = config_file or os.environ.get('CONFIG_FILE') or CONFIG_FILE
    if not os.path.exists(config_file):
        raise FileNotFoundError(f'配置文件不存在: {config_file}')

    ext = os.path.splitext(config_file)[1].lower()
    with open(config_file, 'r', encoding='utf-8') as f:
        if ext in ('.yaml', '.yml'):
            data = yaml.safe_load(f) or {}
        elif ext in ('.json', '.json5'):
            data = json.load(f) or {}
        else:
            raise ValueError(f'不支持的文件格式: {ext}')

    print(f'加载配置: {config_file}')
    return data


def write_config(data, file_path):
    """写入配置文件"""
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, 'w', encoding='utf-8') as f:
        if ext in ('.yaml', '.yml'):
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
        elif ext in ('.json', '.json5'):
            json.dump(data, f, indent=4)
        else:
            raise ValueError(f'不支持的文件格式: {ext}')


if __name__ == '__main__':
    config = load_config()
    print(config)
