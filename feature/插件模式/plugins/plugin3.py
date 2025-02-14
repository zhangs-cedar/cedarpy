from .base_plugin import PluginBase


class Plugin3(PluginBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, vi_item):
        """执行器"""
        print(f"{self.name} 执行器")
