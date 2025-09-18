from abc import ABC, abstractmethod


class PluginBase(ABC):
    """插件基类"""

    def __init__(self, plugin_config, all_config=None):
        """
        :param plugin_config: 插件配置
        :param all_config: 所有配置,用于获取全局配置 非必要不建议直接使用全局配置
        """
        self.plugin_config = plugin_config
        self.all_config = all_config
        self.init_tag = False  # 每个插件类内初始化一次的标志
        self.plugin_name = self.plugin_config.get('plugin_name')

    @abstractmethod
    def init(self):
        """插件初始化"""
        self.init_tag = True
        pass

    @abstractmethod
    def execute(self, data):
        """插件执行体"""
        if not self.init_tag:
            self.init()
        print('[Plugin]: {} execute!'.format(self.plugin_name))
        # 执行 data 处理
        return data
