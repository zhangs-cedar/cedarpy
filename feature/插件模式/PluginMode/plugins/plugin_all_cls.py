# 使用的插件需要全部导入
from .test.plugin1 import Plugin1
from .test.plugin2 import Plugin2
from .test.plugin3 import Plugin3

# print(f"loaded packages: {[name for name in globals() if not name.startswith('_')]}")
