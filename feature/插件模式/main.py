import json
from plugins.all_plugins import *


class PluginManager:
    def __init__(self, config_path="config.json", all_config=None):
        """
        初始化插件管理器
        """
        self.config_path = config_path
        self.plugins_map = self.init_plugins()
        self.all_config = all_config

    def init_plugins(self):
        """
        加载配置文件,并按优先级排序,返回一个字典,key为插件名,value为插件实例
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            plugin_configs = json.load(f)
        # 按优先级排序
        plugin_configs.sort(key=lambda x: x["priority"], reverse=True)
        # 挑选enabled为true的插件
        plugin_configs = [x for x in plugin_configs if x["enabled"]]
        # 初始化所有插件
        plugins_map = {}
        for config in plugin_configs:
            print(f"Initializing plugin: {config['name']}")
            plugins_map[config["name"]] = eval(config["name"])(plugin_config=config, all_config=self.all_config)
        return plugins_map

    def execute_all_plugins(self, data):
        """
        执行所有插件
        """
        for plugin in self.plugins_map:
            print(f"Executing plugin: {plugin.plugin_config['name']}")
            data = plugin.execute(data)
            print(f"Plugin output: {data}")


# 使用示例
if __name__ == "__main__":
    config_path = "/Users/zhangsong/workspace/OpenSource/cedarpy/feature/插件模式/config.json"
    plugin_manager = PluginManager(config_path)
    data = {
        "title": "test",
        "content": "test content",
        "author": "zhangsong",
        "tags": ["python", "open source"],
        "date": "2023-01-01",
        "url": "https://www.baidu.com",
        "source": "baidu",
        "description": "test description",
        "image": "https://www.baidu.com/img/bd_logo1.png",
    }
    plugin_manager.execute_all_plugins(data)
