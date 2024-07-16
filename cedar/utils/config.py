import yaml
import json
import os
from cedar.utils.dict2obj import Dict2Obj


class Config:
    def __init__(self, config_path):
        """初始化配置类."""
        self.config_data = self.load(config_path)  # 加载配置文件（可能是YAML或JSON）
        self.obj = Dict2Obj(self.config_data)  # 将字典转换为对象
        # 输出配置文件的加载路径，但不输出具体的配置内容
        print("配置文件加载解析成功！\npath: {}".format(config_path))

    def load(self, path):
        """根据文件路径的扩展名来判断是加载 YAML 还是 JSON 文件"""
        _, extension = os.path.splitext(path)
        if extension == ".yaml" or extension == ".yml":
            return self.read_yaml(path)
        elif extension == ".json":
            return self.read_json(path)
        else:
            raise ValueError("Unsupported file format: {}".format(extension))

    @staticmethod
    def read_yaml(path):
        """ """
        with open(path, encoding="utf-8") as file:
            json = yaml.safe_load(file)  # 加载YAML数据
        return json

    # 读取 JSON 文件
    @staticmethod
    def read_json(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

    # 修改 JSON 文件
    @staticmethod
    def write_json(data, file_path):
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
