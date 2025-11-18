from plugins.plugin_base import PluginBase
from typing import Any


class Plugin2(PluginBase):
    def init(self):
        """插件初始化"""
        print(f'Plugin2 初始化完成')

    def execute(self, data: Any) -> Any:
        """插件执行体"""
        print(f'[{self.plugin_name}] 处理数据: {data}')
        if isinstance(data, dict):
            data['processed_by'] = data.get('processed_by', []) + ['Plugin2']
        return data
