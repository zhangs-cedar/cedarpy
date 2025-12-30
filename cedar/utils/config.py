import os
import json
import json5
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
        elif ext == '.json':
            data = json.load(f) or {}
        elif ext == '.json5':
            data = json5.load(f) or {}
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
        elif ext == '.json':
            json.dump(data, f, indent=4, ensure_ascii=False)
        elif ext == '.json5':
            json5.dump(data, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError(f'不支持的文件格式: {ext}')


if __name__ == '__main__':
    # 读取配置示例
    config = load_config()
    print('读取配置:', config)
    
    # 写入配置示例
    test_data = {
        'app': {
            'name': 'CedarPy',
            'version': '1.0.0'
        },
        'database': {
            'host': 'localhost',
            'port': 3306
        }
    }
    
    # 写入 YAML 格式
    write_config(test_data, 'test_config.yaml')
    print('已写入: test_config.yaml')
    
    # 写入 JSON 格式（标准 JSON）
    write_config(test_data, 'test_config.json')
    print('已写入: test_config.json')
    
    # 写入 JSON5 格式
    write_config(test_data, 'test_config.json5')
    print('已写入: test_config.json5')
    
    # 验证写入的文件
    loaded_yaml = load_config('test_config.yaml')
    print('验证 YAML:', loaded_yaml)
    
    loaded_json = load_config('test_config.json')
    print('验证 JSON:', loaded_json)
    
    loaded_json5 = load_config('test_config.json5')
    print('验证 JSON5:', loaded_json5)
