from .base_plugin import PluginBase


class Plugin1(PluginBase):
    def __init__(self, *args, **kwargs):
        self.name = "Plugin1"

    def execute(self, vi_item):
        """执行器"""
        print(f"{self.name} 执行器")
