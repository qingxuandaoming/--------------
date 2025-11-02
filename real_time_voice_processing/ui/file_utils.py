#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频文件工具

提供默认测试目录与音频文件收集函数，供 UI 复用。
"""

from __future__ import annotations

import os
from typing import List

from real_time_voice_processing.runtime.audio_source import SUPPORTED_EXTENSIONS


def default_audio_dir() -> str:
    """
    获取默认音频目录路径。

    Returns
    -------
    str
        默认音频测试目录的绝对路径（`real_time_voice_processing/assets/audio_tests`）。
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(pkg_dir, "assets", "audio_tests")
    os.makedirs(d, exist_ok=True)
    return d


def collect_audio_files(directory: str) -> List[str]:
    """
    收集目录下支持的音频文件列表。

    Parameters
    ----------
    directory : str
        目标目录路径。

    Returns
    -------
    list of str
        按文件名排序的音频文件绝对路径列表。
    """
    exts = {e.lower() for e in SUPPORTED_EXTENSIONS}
    files: List[str] = []
    if not os.path.isdir(directory):
        return files
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in exts:
            files.append(path)
    return files