import json
from .plugin_all_cls import *


class PluginManager:
    """
    # 方式1：通过配置文件路径初始化
    config_path = "config.json"
    plugin_manager = PluginManager(config_path=config_path)
    data = {"title": "test", "content": "test content"}
    result = plugin_manager.execute_all_plugins(data)
    print(result)

    # 方式2：通过已解析好的配置数据初始化
    config_data = [
        {
            "name": "Plugin1",
            "__name__": "插件名称",
            "enabled": True,
            "__enabled__": "是否启用插件",
            "priority": 99,
            "__priority__": "插件优先级",
            "config": {"123": 123},
            "__config__": "插件配置信息"
        },
        {
            "name": "Plugin2",
            "enabled": False,
            "priority": 99,
            "config": {"123": 123}
        },
        {
            "name": "Plugin3",
            "enabled": True,
            "priority": 100,
            "config": {"123": 123}
        }
    ]
    plugin_manager = PluginManager(config_data=config_data)
    data = {"title": "test", "content": "test content"}
    result = plugin_manager.execute_all_plugins(data)
    print(result)
    """

    def __init__(self, config_path=None, config_data=None, all_config=None):
        """
        初始化插件管理器
        :param config_path: 配置文件路径（可选）
        :param config_data: 已解析好的配置数据（可选）
        :param all_config: 全局配置（可选）
        """
        self.config_path = config_path
        self.config_data = config_data
        self.all_config = all_config
        self.plugins_map = self.init_plugins()

    def load_config(self):
        """
        加载配置文件或直接使用已解析好的配置数据
        :return: 配置数据
        """
        if self.config_data:
            # 如果直接传入了配置数据，则直接使用
            return self.config_data
        elif self.config_path:
            # 如果传入了配置文件路径，则从文件加载
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError('Either config_path or config_data must be provided')

    def init_plugins(self):
        """
        加载配置文件,并按优先级排序,返回一个字典,key为插件名,value为插件实例
        """
        plugin_configs = self.load_config()
        # 按优先级排序
        plugin_configs.sort(key=lambda x: x['priority'], reverse=True)
        # 挑选enabled为true的插件
        plugin_configs = [x for x in plugin_configs if x['enabled']]
        # 初始化所有插件
        plugins_map = {}
        for config in plugin_configs:
            # print(f"[Initializing plugin]: {config['name']}")
            plugins_map[config['name']] = eval(config['name'])(plugin_config=config, all_config=self.all_config)
        return plugins_map

    def execute_all_plugins(self, data):
        """
        执行所有插件
        TODO: data 是传入的对象,结构未定,后续最好抽象成固定结构的对象。
        """
        for plugin_name, plugin in self.plugins_map.items():
            # print(f"[Executing] plugin: {plugin_name}")
            data = plugin.execute(data)
        return data
