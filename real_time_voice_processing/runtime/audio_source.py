#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频源接口抽象（AudioSource）

提供统一的音频数据读取接口，便于在运行时引擎中替换底层采集实现：
 - PyAudioSource：从系统麦克风实时读取
 - FileAudioSource：从音频文件顺序读取（使用 soundfile，可选依赖）

接口约定：
 - open()：打开资源
 - read(num_frames) -> numpy.ndarray[int16]：读取指定样本数；如到达 EOF 可返回空数组
 - close()：关闭资源
 - sample_rate / channels：属性，指示采样率与声道数
"""

from __future__ import annotations

from typing import Optional
import numpy as np


class AudioSource:
    """音频源基类接口。"""

    sample_rate: int
    channels: int

    def open(self) -> None:
        raise NotImplementedError

    def read(self, num_frames: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class PyAudioSource(AudioSource):
    """
    基于 PyAudio 的音频源实现。

    Notes
    -----
    - 依赖 `pyaudio` 包；若未安装或设备不可用，打开时会抛出异常。
    - 返回的数组类型为 `int16`，与引擎处理链一致。
    """

    def __init__(self, sample_rate: int, channels: int, format_const: int, frames_per_buffer: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._format = format_const
        self._fpb = frames_per_buffer
        self._pyaudio = None
        self._stream = None

    def open(self) -> None:
        import pyaudio  # 局部导入，降低模块级依赖

        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=self._format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self._fpb,
        )

    def read(self, num_frames: int) -> np.ndarray:
        assert self._stream is not None, "PyAudioSource 未打开"
        data = self._stream.read(num_frames, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def close(self) -> None:
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
        finally:
            if self._pyaudio:
                self._pyaudio.terminate()
            self._stream = None
            self._pyaudio = None


class FileAudioSource(AudioSource):
    """
    基于 SoundFile 的文件音频源实现（可选依赖）。

    Parameters
    ----------
    file_path : str
        输入音频文件路径。
    sample_rate : int | None
        若提供则在读取后执行简单重采样（最近邻），用于与运行时处理链对齐。
        若为 None 则不重采样，直接返回文件采样率。

    Notes
    -----
    - 依赖 `soundfile` 包；若未安装会在 open() 时抛出异常。
    - 返回的数组类型为 `int16`。
    - 简易重采样仅为测试/演示目的，生产环境建议使用高质量重采样（如 librosa）。
    """

    def __init__(self, file_path: str, sample_rate: Optional[int] = None) -> None:
        self._file_path = file_path
        self._sf = None
        self._target_sr = sample_rate
        self.sample_rate = 0
        self.channels = 0

    def open(self) -> None:
        import soundfile as sf  # 局部导入，降低模块级依赖

        self._sf = sf.SoundFile(self._file_path, mode="r")
        self.sample_rate = int(self._sf.samplerate)
        self.channels = int(self._sf.channels)

    def read(self, num_frames: int) -> np.ndarray:
        assert self._sf is not None, "FileAudioSource 未打开"
        import soundfile as sf

        # 读取为 int16
        data = self._sf.read(num_frames, dtype="int16", always_2d=False)
        if data is None:
            return np.array([], dtype=np.int16)
        arr = np.array(data, dtype=np.int16)

        # 若指定目标采样率且不同，执行简易重采样（最近邻）
        if self._target_sr and self._target_sr != self.sample_rate:
            src_len = len(arr)
            if src_len == 0:
                return arr
            ratio = self._target_sr / float(self.sample_rate)
            new_len = max(1, int(src_len * ratio))
            idx = (np.arange(new_len) / ratio).astype(np.int64)
            idx = np.clip(idx, 0, src_len - 1)
            arr = arr[idx]
        return arr

    def close(self) -> None:
        try:
            if self._sf:
                self._sf.close()
        finally:
            self._sf = None