class Dict2Obj(dict):
    """字典转对象，可以通过属性访问,支持嵌套,但是不支持修改!

    Example:
        >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> d2o = Dict2Obj(d)
        >>> d2o.c.d
        >>> 3

    """

    def __init__(self, *args, **kwargs):
        super(Dict2Obj, self).__init__(*args, **kwargs)

    def __getattr__(self, key: dict) -> object:
        value = self[key]
        if isinstance(value, dict):
            value = Dict2Obj(value)
        return value
