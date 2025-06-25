import yaml
import json5 as json
import os
from typing import Any, Dict, Union
from cedar.utils.dict2obj import Dict2Obj


class Config:
    """配置管理类，支持加载YAML和JSON和JSON5格式的配置文件"""

    def __init__(self, config_path: str) -> None:
        """初始化配置类

        Args:
            config_path: 配置文件路径

        Raises:
            ValueError: 当文件格式不支持时
        """
        self.config_data = self.load(config_path)
        self.obj = Dict2Obj(self.config_data)
        print(f"配置文件加载解析成功！\npath: {config_path}")

    def load(self, path: str) -> Dict[str, Any]:
        """根据文件扩展名加载配置文件

        Args:
            path: 配置文件路径

        Returns:
            Dict[str, Any]: 配置数据字典

        Raises:
            ValueError: 当文件格式不支持时
        """
        _, extension = os.path.splitext(path)
        if extension in [".yaml", ".yml"]:
            return self.read_yaml(path)
        elif extension in [".json", ".json5"]:
            return self.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    @staticmethod
    def read_yaml(path: str) -> Dict[str, Any]:
        """读取YAML配置文件

        Args:
            path: YAML文件路径

        Returns:
            Dict[str, Any]: 解析后的配置数据
        """
        with open(path, encoding="utf-8") as file:
            return yaml.safe_load(file)

    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        """读取JSON配置文件

        Args:
            file_path: JSON文件路径

        Returns:
            Dict[str, Any]: 解析后的配置数据
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def write_json(data: Dict[str, Any], file_path: str) -> None:
        """写入JSON配置文件

        Args:
            data: 要写入的数据
            file_path: 目标文件路径
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
