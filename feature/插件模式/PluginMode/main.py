import json
from abc import ABC, abstractmethod
from plugins.plugin_manager import PluginManager


# 使用示例
if __name__ == "__main__":
    config_path = "/Users/zhangsong/workspace/OpenSource/cedarpy/feature/插件模式/config.json"
    plugin_manager = PluginManager(config_path)
    data = {
        "title": "test",
        "content": "test content",
        "author": "zhangsong",
        "tags": ["python", "open source"],
        "date": "2023-01-01",
        "url": "https://www.baidu.com",
        "source": "baidu",
        "description": "test description",
        "image": "https://www.baidu.com/img/bd_logo1.png",
    }
    plugin_manager.execute_all_plugins(data)
