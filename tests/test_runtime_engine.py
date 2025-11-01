#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np

from real_time_voice_processing.runtime.engine import AudioRuntime
from real_time_voice_processing.runtime.audio_source import AudioSource
from real_time_voice_processing.config import Config


class DummyAudioSource(AudioSource):
    """
    测试用音频源：生成固定长度的正弦波并按需读取。
    返回 int16 数据，采样率与声道数与 Config 对齐。
    """

    def __init__(self, duration_sec: float = 0.25, freq_hz: float = 440.0) -> None:
        self.sample_rate = Config.SAMPLE_RATE
        self.channels = 1
        t = np.arange(int(duration_sec * self.sample_rate)) / float(self.sample_rate)
        x = 0.5 * np.sin(2 * np.pi * freq_hz * t)
        self._data = (x * 32767).astype(np.int16)
        self._pos = 0

    def open(self) -> None:
        self._pos = 0

    def read(self, num_frames: int) -> np.ndarray:
        if self._pos >= len(self._data):
            return np.array([], dtype=np.int16)
        end = min(len(self._data), self._pos + num_frames)
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk

    def close(self) -> None:
        pass


def test_runtime_engine_with_dummy_source():
    # 使用 DummyAudioSource 驱动引擎，验证处理线程产出
    src = DummyAudioSource(duration_sec=0.3)
    rt = AudioRuntime(audio_source=src)
    rt.start()
    # 等待一小段时间以触发处理
    time.sleep(0.05)
    rt.stop()

    energies, zcrs, vads = rt.get_recent_processed()
    assert energies.size > 0, "应产生至少一帧能量数据"
    assert zcrs.size == energies.size, "能量与过零率帧数应一致"
    assert vads.size == energies.size, "VAD 帧数应一致"