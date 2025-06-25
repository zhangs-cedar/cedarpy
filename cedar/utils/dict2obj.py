from typing import Any, Dict


class Dict2Obj:
    """字典转对象类，支持嵌套、访问对象属性、修改属性值、修改嵌套属性值、创建新属性"""

    def __init__(self, dictionary: Dict[str, Any]) -> None:
        """初始化方法，将字典的键值对转换为对象的属性

        Args:
            dictionary: 要转换的字典
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Obj(value))
            else:
                setattr(self, key, value)
