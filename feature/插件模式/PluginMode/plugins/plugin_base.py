from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PluginBase(ABC):
    """插件基类 - 简洁高效"""

    def __init__(self, plugin_config: Dict[str, Any]):
        """
        :param plugin_config: 插件配置
        """
        self.plugin_config = plugin_config
        self.plugin_name = plugin_config.get('plugin_name', self.__class__.__name__)
        self.config = plugin_config.get('config', {})
        self._initialized = False
        
        # 构造时就初始化，别搞什么懒加载
        self._init()

    def _init(self):
        """内部初始化，只执行一次"""
        if not self._initialized:
            self.init()
            self._initialized = True

    @abstractmethod
    def init(self):
        """插件初始化 - 子类实现"""
        pass

    @abstractmethod
    def execute(self, data: Any) -> Any:
        """插件执行 - 子类实现"""
        pass

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.plugin_name})"
