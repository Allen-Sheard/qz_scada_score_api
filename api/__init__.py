# -*- coding: utf-8 -*-
"""
api 包：封装 QZ、SCADA 检测与打分功能，供外部程序调用。

用法示例：
    from api import QZAnalyzer, SCADAAnalyzer, ScoreEngine

    qz = QZAnalyzer()
    result = qz.analyze("/path/to/qz.jpg")
"""

import os

# 部分环境 protobuf 版本与 onnx 不兼容，设置此项可绕过版本检查。
# 该变量需在导入 ultralytics/onnx 前设置。
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from api.interface_qz import QZAnalyzer
from api.interface_scada import SCADAAnalyzer
from api.interface_score import ScoreEngine

__all__ = ["QZAnalyzer", "SCADAAnalyzer", "ScoreEngine"]
