from plugins import PluginManager


if __name__ == '__main__':
    # 初始化插件管理器
    manager = PluginManager(config_path='config.json')
    
    # 列出可用插件
    print("=== 可用插件 ===")
    plugins_info = manager.list_plugins()
    for stage, plugins in plugins_info.items():
        print(f"{stage}: {plugins}")
    
    # 测试数据
    data = {'title': 'test', 'content': 'test content', 'processed_by': []}
    print(f"\n=== 初始数据 ===\n{data}")
    
    # 执行所有阶段
    print(f"\n=== 执行所有阶段 ===")
    result = manager.execute_all(data)
    print(f"\n=== 最终结果 ===\n{result}")
