class Dict2Obj:
    """
    字典转对象 支持嵌套、访问对象属性、修改属性值、修改嵌套属性值、创建新属性

    """

    def __init__(self, dictionary):
        """
        初始化方法，将字典的键值对转换为对象的属性。
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Obj(value))
            else:
                setattr(self, key, value)
