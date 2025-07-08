from plugins.plugin_manager import PluginManager


# 使用示例
if __name__ == '__main__':
    # 方式1：通过配置文件路径初始化
    config_path = '/Users/zhangsong/workspace/OpenSource/cedarpy/feature/插件模式/PluginMode/config.json'
    plugin_manager = PluginManager(config_path=config_path)
    data = {'title': 'test', 'content': 'test content'}
    result = plugin_manager.execute_all_plugins(data)
    print(result)

    # 方式2：通过已解析好的配置数据初始化
    # config_data = [
    #     {
    #         "name": "Plugin1",
    #         "__name__": "插件名称",
    #         "enabled": True,
    #         "__enabled__": "是否启用插件",
    #         "priority": 99,
    #         "__priority__": "插件优先级",
    #         "config": {"123": 123},
    #         "__config__": "插件配置信息",
    #     },
    #     {"name": "Plugin2", "enabled": False, "priority": 99, "config": {"123": 123}},
    #     {"name": "Plugin3", "enabled": True, "priority": 100, "config": {"123": 123}},
    # ]
    # plugin_manager = PluginManager(config_data=config_data)
    # data = {"title": "test", "content": "test content"}
    # result = plugin_manager.execute_all_plugins(data)
    # print(result)
