# GlobalDataIndexSGZ.py
# 兼容层：SGZ 模式自动固定为单空格分隔，保持原有导入方式不变

from common.global_data_index import GlobalDataIndex as _GlobalDataIndex

class GlobalDataIndex(_GlobalDataIndex):
    """
    SGZ 专用兼容类，继承自统一的 GlobalDataIndex。
    默认 mode='sgz'，其他参数与父类完全一致。
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mode", "sgz")
        super().__init__(*args, **kwargs)
