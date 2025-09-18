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
            return self.config_data
        elif self.config_path:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(
                'Either config_path or config_data must be provided')

    def init_plugins(self):
        """
        加载配置文件,按阶段组织插件实例
        :return: 按阶段组织的插件字典 {stage: [(priority, plugin_instance), ...]}
        """
        config = self.load_config()
        plugins_map = {}

        for stage, plugin_configs in config.items():
            stage_plugins = []
            # 按优先级排序
            plugin_configs.sort(key=lambda x: x['priority'], reverse=True)
            # 只处理enabled的插件
            enabled_configs = [x for x in plugin_configs if x['enabled']]

            for plugin_config in enabled_configs:
                plugin_name = plugin_config['name']
                plugin_instance = eval(plugin_name)(
                    plugin_config=plugin_config, all_config=self.all_config)
                stage_plugins.append(
                    (plugin_config['priority'], plugin_instance))

            plugins_map[stage] = stage_plugins

        print(plugins_map)

        return plugins_map

    def execute_all_plugins(self, data):
        """
        按顺序执行所有阶段的插件
        """
        for stage in self.plugins_map.keys():
            for priority, plugin in self.plugins_map[stage]:
                print(f"执行阶段: {stage}, 插件: {plugin.name}, 优先级: {priority}")
                data = plugin.execute(data)
        return data
