import json5 as json
import os

import yaml

from cedar.utils.dict2obj import Dict2Obj


class Config:
    """配置管理类"""

    def __init__(self, config_path):
        self.config_data = self.load(config_path)
        self.obj = Dict2Obj(self.config_data)
        print(f'加载配置: {config_path}')

    def load(self, path):
        """加载配置文件"""
        ext = os.path.splitext(path)[1].lower()

        with open(path, 'r', encoding='utf-8') as f:
            if ext in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            elif ext in ('.json', '.json5'):
                return json.load(f)
            else:
                raise ValueError(f'不支持的文件格式: {ext}')

    @staticmethod
    def write_json(data, file_path):
        """写入JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
