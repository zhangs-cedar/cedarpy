import json
import os
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from plugins.plugin_base import PluginBase


class PluginManager:
    """插件管理器"""

    def __init__(self, config_path: Optional[str] = None, 
                 config_data: Optional[Dict] = None,
                 plugin_dirs: Optional[List[str]] = None):
        """
        初始化插件管理器
        :param config_path: 配置文件路径
        :param config_data: 配置数据
        :param plugin_dirs: 插件目录列表
        """
        self.config_path = config_path
        self.config_data = config_data
        self.plugin_dirs = plugin_dirs or [
            os.path.join(os.path.dirname(__file__), 'test'),
            os.path.join(os.path.dirname(__file__), 'test2')
        ]
        self.registered_plugins = {}  # {plugin_name: plugin_class}
        self.stage_plugins = {}  # {stage: [(priority, plugin_instance)]}
        
        self._discover_plugins()
        self._init_plugins()

    def _discover_plugins(self):
        """动态发现插件类"""
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
                
            for file_path in Path(plugin_dir).glob('*.py'):
                if file_path.name.startswith('__'):
                    continue
                    
                try:
                    # 构建模块名：test.plugin1 这样的形式
                    module_name = f"{Path(plugin_dir).name}.{file_path.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 查找插件类 - 简化检测逻辑
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if self._is_plugin_class(obj):
                            self.registered_plugins[name] = obj
                            
                except Exception as e:
                    print(f"警告: 加载插件 {file_path} 失败: {e}")
    
    def _is_plugin_class(self, cls) -> bool:
        """检查是否为有效插件类"""
        return (issubclass(cls, PluginBase) and 
                cls != PluginBase and
                hasattr(cls, 'execute') and 
                hasattr(cls, 'init'))

    def _load_config(self) -> Dict:
        """加载配置"""
        if self.config_data:
            return self.config_data
        elif self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"配置加载失败: {e}")
                return {}
        return {}

    def _init_plugins(self):
        """初始化插件实例"""
        config = self._load_config()
        
        for stage, plugin_configs in config.items():
            stage_plugins = []
            
            for plugin_config in plugin_configs:
                if not plugin_config.get('enabled', True):
                    continue
                    
                plugin_name = plugin_config.get('plugin_name')
                if not plugin_name or plugin_name not in self.registered_plugins:
                    print(f"警告: 插件 {plugin_name} 未找到")
                    continue
                
                try:
                    plugin_class = self.registered_plugins[plugin_name]
                    plugin_instance = plugin_class(plugin_config)
                    priority = plugin_config.get('priority', 0)
                    stage_plugins.append((priority, plugin_instance))
                except Exception as e:
                    print(f"警告: 插件 {plugin_name} 初始化失败: {e}")
            
            # 按优先级排序
            stage_plugins.sort(key=lambda x: x[0], reverse=True)
            self.stage_plugins[stage] = stage_plugins

    def execute_stage(self, stage: str, data: Any) -> Any:
        """执行指定阶段的插件"""
        if stage not in self.stage_plugins:
            return data
            
        for priority, plugin in self.stage_plugins[stage]:
            try:
                print(f"执行: {stage}.{plugin.plugin_name} (优先级:{priority})")
                data = plugin.execute(data)
            except Exception as e:
                print(f"错误: 插件 {plugin.plugin_name} 执行失败: {e}")
                # 继续执行其他插件，不中断流程
        return data

    def execute_all(self, data: Any) -> Any:
        """执行所有阶段插件"""
        for stage in self.stage_plugins:
            data = self.execute_stage(stage, data)
        return data

    def list_plugins(self) -> Dict[str, List[str]]:
        """列出所有可用插件"""
        result = {'registered': list(self.registered_plugins.keys())}
        for stage, plugins in self.stage_plugins.items():
            result[stage] = [p[1].plugin_name for p in plugins]
        return result
