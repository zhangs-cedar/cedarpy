import logging
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

    @abstractmethod
    def execute(self, data):
        """插件执行体"""
