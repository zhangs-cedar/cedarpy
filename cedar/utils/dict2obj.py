class Dict2Obj:
    """字典转对象，支持嵌套"""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, Dict2Obj(v) if isinstance(v, dict) else v)
