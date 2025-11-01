"""
real_time_voice_processing 包
用于实时语音信号处理课程设计的核心代码模块。

建议通过包导入使用：
    from real_time_voice_processing.signal_processing import SignalProcessing
    from real_time_voice_processing.config import Config
"""

__all__ = [
    "Config",
    "SignalProcessing",
]

from .config import Config
from .signal_processing import SignalProcessing