from plugins.plugin_manager import PluginManager


# 使用示例
if __name__ == '__main__':
    # 通过配置文件初始化
    config_path = 'config.json'
    plugin_manager = PluginManager(config_path=config_path)
    data = {'title': 'test', 'content': 'test content'}
    
    # 执行所有阶段
    print("=== 执行所有阶段 ===")
    result = plugin_manager.execute_all_plugins(data)
    print(f"最终结果: {result}")
