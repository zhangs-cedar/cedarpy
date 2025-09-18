from ..plugin_base import PluginBase


class Plugin2(PluginBase):
    def init(self):
        """插件初始化"""
        self.init_tag = True
        pass

    def execute(self, data):
        """插件执行体"""
        if not self.init_tag:
            self.init()
        print('[Plugin]: {} execute!'.format(self.plugin_name))
        # 执行 data 处理
        return data
