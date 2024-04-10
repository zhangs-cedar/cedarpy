import yaml

from cedar.utils.dict2obj import Dict2Obj


class ConfigYaml:
    """配置项的值是否存在,如果配置不存在,返回defval

    Example:
        >>> config = ConfigYaml(osp.join(base_dir, config_file))
        >>> print(config._cache)
        >>> {'A': {'B': '1'}}
        >>> print(config.get_config("A.B", defval="0"))
        >>> '1'
    """

    def __init__(self, config_path):
        """初始化配置类."""
        self.json = self.load_yaml(config_path)  # 加载YAML文件
        self.obj = self.to_obj(self.json)  # 将字典转换为对象
        print("YAML文件加载解析成功！\npath: {} \nyaml: {}".format(config_path, self.json))

    def load_yaml(self, path):
        """
        加载YAML文件并解析为Python字典。

        参数：
        - path：YAML文件的路径

        返回：
        - json：解析后的Python字典
        """
        with open(path, encoding="utf-8") as file:
            json = yaml.safe_load(file)  # 加载YAML数据
        return json

    def to_obj(self, json):
        """
        将Python字典转换为对象。

        参数：
        - json：Python字典

        返回：
        - obj：转换后的对象
        """
        obj = Dict2Obj(json)  # 将字典转换为对象
        return obj
