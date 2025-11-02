#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
离线文件统计工具

读取音频文件并按与运行时一致的处理链（分帧、加窗、特征、复合VAD+平滑）
计算语音占比与关键统计（能量、ZCR、谱熵）。

使用方法（Windows PowerShell）：

py -3.10 -m real_time_voice_processing.analyze_file \
    --file .\real_time_voice_processing\assets\audio_tests\20250903_145702.m4a

可选参数：
  --sample-rate <int>      重采样目标采样率（默认使用 Config.SAMPLE_RATE）
  --json                   以 JSON 输出结果便于机器处理
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from typing import Any, Dict, List

import numpy as np

from .config import Config
from .signal_processing import SignalProcessing
from .runtime.audio_source import FileAudioSource


def compute_file_stats(file_path: str, sample_rate: int | None = None) -> Dict[str, Any]:
    """读取文件并计算语音占比与能量/ZCR/谱熵统计。"""
    src = FileAudioSource(file_path, sample_rate=sample_rate or Config.SAMPLE_RATE)
    src.open()
    try:
        # 读取整文件到数组
        chunks: List[np.ndarray] = []
        while True:
            buf = src.read(Config.CHUNK_SIZE)
            if buf is None or len(buf) == 0:
                break
            chunks.append(np.array(buf, dtype=np.int16, copy=False))
        if len(chunks) == 0:
            pcm = np.array([], dtype=np.int16)
        else:
            pcm = np.concatenate(chunks)

        rate = int(src.sample_rate or (sample_rate or Config.SAMPLE_RATE))
        frame_size = Config.FRAME_SIZE
        hop_size = Config.HOP_SIZE
        window = SignalProcessing.hamming_window(frame_size)

        # 分帧并计算特征/判决（与运行时一致）
        frames = SignalProcessing.framing(pcm.astype(np.float32), frame_size, hop_size)
        energies: List[float] = []
        zcrs: List[float] = []
        entropies: List[float] = []
        vads: List[int] = []

        energy_history: deque[float] = deque(maxlen=256)
        zcr_history: deque[float] = deque(maxlen=256)
        vad_hold = 0
        silence_run = 0

        for fr in frames:
            frame = (fr * window).astype(np.float32)
            energy = float(SignalProcessing.calculate_short_time_energy(frame))
            zcr = float(SignalProcessing.calculate_zero_crossing_rate(frame))
            entropy = float(
                SignalProcessing.calculate_spectral_entropy(frame, n_fft=Config.SPECTRAL_ENTROPY_N_FFT)
            )

            energy_gate = bool(energy > Config.ENERGY_THRESHOLD)
            zcr_gate = bool(zcr < Config.ZCR_THRESHOLD)
            entropy_gate = bool(entropy < Config.SPECTRAL_ENTROPY_VOICE_MAX)
            vad_initial = bool(energy_gate and (zcr_gate or entropy_gate))

            vad_adapt = SignalProcessing.adaptive_voice_activity_detection(
                energy,
                zcr,
                list(energy_history),
                list(zcr_history),
                energy_k=Config.ADAPTIVE_VAD_ENERGY_K,
                zcr_k=Config.ADAPTIVE_VAD_ZCR_K,
                min_history=Config.ADAPTIVE_VAD_HISTORY_MIN,
                fallback_energy_threshold=Config.ENERGY_THRESHOLD,
                fallback_zcr_threshold=Config.ZCR_THRESHOLD,
            )
            if Config.USE_ADAPTIVE_VAD:
                vad_initial = bool(vad_initial or bool(vad_adapt))

            if vad_initial:
                vad_hold = max(vad_hold, int(Config.VAD_HANGOVER_ON))
                silence_run = 0
                vad = 1
            else:
                if vad_hold > 0:
                    vad_hold -= 1
                    vad = 1
                    silence_run = 0
                else:
                    silence_run += 1
                    vad = 0 if silence_run >= int(Config.VAD_RELEASE_OFF) else 1

            energies.append(energy)
            zcrs.append(zcr)
            entropies.append(entropy)
            vads.append(int(vad))
            energy_history.append(energy)
            zcr_history.append(zcr)

        # 统计
        total_frames = len(vads)
        voice_frames = int(np.sum(vads))
        voice_ratio = (voice_frames / total_frames) if total_frames > 0 else 0.0

        result = {
            "file": file_path,
            "sample_rate": rate,
            "frames": total_frames,
            "voice_frames": voice_frames,
            "voice_ratio": voice_ratio,
            "energy": {
                "mean": float(np.mean(energies)) if total_frames > 0 else 0.0,
                "min": float(np.min(energies)) if total_frames > 0 else 0.0,
                "max": float(np.max(energies)) if total_frames > 0 else 0.0,
            },
            "zcr": {
                "mean": float(np.mean(zcrs)) if total_frames > 0 else 0.0,
                "min": float(np.min(zcrs)) if total_frames > 0 else 0.0,
                "max": float(np.max(zcrs)) if total_frames > 0 else 0.0,
            },
            "spectral_entropy": {
                "mean": float(np.mean(entropies)) if total_frames > 0 else 0.0,
                "min": float(np.min(entropies)) if total_frames > 0 else 0.0,
                "max": float(np.max(entropies)) if total_frames > 0 else 0.0,
            },
        }
        return result
    finally:
        try:
            src.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="离线音频文件统计（语音占比与能量/ZCR/谱熵）")
    parser.add_argument("--file", required=False, default=None, help="音频文件路径")
    parser.add_argument("--sample-rate", type=int, default=Config.SAMPLE_RATE, help="重采样目标采样率")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args()

    file_path = args.file
    if not file_path:
        # 默认使用示例库中的 20 分钟音频
        import os
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "audio_tests", "20250903_145702.m4a"
        )

    stats = compute_file_stats(file_path, sample_rate=args.sample_rate)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print(f"文件: {stats['file']}")
        print(f"采样率: {stats['sample_rate']} Hz")
        print(f"总帧数: {stats['frames']}")
        print(f"语音帧数: {stats['voice_frames']}")
        print(f"语音占比: {stats['voice_ratio']*100:.2f}%")
        e = stats["energy"]
        z = stats["zcr"]
        se = stats["spectral_entropy"]
        print(f"能量: mean {e['mean']:.1f}, min {e['min']:.1f}, max {e['max']:.1f}")
        print(f"ZCR: mean {z['mean']:.4f}, min {z['min']:.4f}, max {z['max']:.4f}")
        print(f"谱熵: mean {se['mean']:.4f}, min {se['min']:.4f}, max {se['max']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()