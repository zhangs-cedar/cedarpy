from plugins.plugin_base import PluginBase
from typing import Any


class Plugin1(PluginBase):
    def init(self):
        """插件初始化"""
        print(f"Plugin1 初始化完成")

    def execute(self, data: Any) -> Any:
        """插件执行体"""
        print(f'[{self.plugin_name}] 处理数据: {data}')
        # 示例：给数据加个标记
        if isinstance(data, dict):
            data['processed_by'] = data.get('processed_by', []) + ['Plugin1']
        return data
