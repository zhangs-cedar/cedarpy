# 使用的插件需要全部导入
from .postprocess.plugin1 import Plugin1
from .postprocess.plugin2 import Plugin2
from .postprocess.plugin3 import Plugin3

# print(f"loaded packages: {[name for name in globals() if not name.startswith('_')]}")
