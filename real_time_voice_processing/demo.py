#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统演示脚本
用于展示实时语音信号处理系统的基本功能
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_processing import SignalProcessing
from config import Config

def demo_signal_processing():
    """演示信号处理功能"""
    print("=" * 60)
    print("实时语音信号处理系统 - 功能演示")
    print("=" * 60)
    
    # 生成测试信号
    duration = 2  # 2秒
    t = np.arange(int(duration * Config.SAMPLE_RATE)) / Config.SAMPLE_RATE
    
    # 创建包含静音、浊音、清音的测试信号
    signal = np.zeros_like(t)
    
    # 静音段 (0-0.5秒)
    # 浊音段 (0.5-1.0秒) - 正弦波模拟
    freq = 100  # 基频
    signal[int(0.5*Config.SAMPLE_RATE):int(1.0*Config.SAMPLE_RATE)] = \
        np.sin(2 * np.pi * freq * t[int(0.5*Config.SAMPLE_RATE):int(1.0*Config.SAMPLE_RATE)]) * 1000
    
    # 清音段 (1.0-1.5秒) - 白噪声模拟
    signal[int(1.0*Config.SAMPLE_RATE):int(1.5*Config.SAMPLE_RATE)] = \
        np.random.randn(int(0.5*Config.SAMPLE_RATE)) * 300
    
    # 静音段 (1.5-2.0秒)
    
    print(f"生成测试信号 - 时长: {duration}秒, 采样率: {Config.SAMPLE_RATE}Hz")
    print("信号包含: 静音 → 浊音 → 清音 → 静音")
    
    # 分帧处理
    frames = SignalProcessing.framing(signal, Config.FRAME_SIZE, Config.HOP_SIZE)
    print(f"\n分帧结果: {len(frames)}帧, 每帧{Config.FRAME_SIZE}样本点")
    
    # 处理每一帧
    print("\n开始处理信号...")
    results = []
    
    start_time = time.time()
    
    for i, frame in enumerate(frames):
        # 计算特征
        energy = SignalProcessing.calculate_short_time_energy(frame)
        zcr = SignalProcessing.calculate_zero_crossing_rate(frame)
        vad = SignalProcessing.voice_activity_detection(energy, zcr, 
                                                      energy_threshold=100000,
                                                      zcr_threshold=0.05)
        
        results.append({
            'frame': i,
            'time': i * Config.HOP_SIZE / Config.SAMPLE_RATE,
            'energy': energy,
            'zcr': zcr,
            'vad': vad
        })
        
        # 每10帧显示一次进度
        if i % 10 == 0 or i == len(frames) - 1:
            progress = (i + 1) / len(frames) * 100
            print(f"处理进度: {progress:.1f}%", end='\r')
    
    processing_time = time.time() - start_time
    print(f"\n处理完成! 耗时: {processing_time:.3f}秒, 实时因子: {processing_time/duration:.3f}")
    
    # 显示结果统计
    print("\n" + "=" * 50)
    print("处理结果统计:")
    print("=" * 50)
    
    total_frames = len(results)
    voice_frames = sum(1 for r in results if r['vad'] == 1)
    silence_frames = total_frames - voice_frames
    
    print(f"总帧数: {total_frames}")
    print(f"语音帧数: {voice_frames} ({voice_frames/total_frames*100:.1f}%)")
    print(f"静音帧数: {silence_frames} ({silence_frames/total_frames*100:.1f}%)")
    
    # 显示典型帧的特征值
    print("\n典型帧特征值:")
    for i in [5, 15, 25, 35]:  # 选择几个代表性的帧
        if i < len(results):
            r = results[i]
            frame_type = "静音" if r['vad'] == 0 else "语音"
            print(f"帧{i:2d} ({r['time']:.2f}s): 能量={r['energy']:.1e}, 过零率={r['zcr']:.3f}, 类型={frame_type}")
    
    # 计算检测准确率 (基于已知的信号结构)
    print("\n检测准确率评估:")
    correct_detections = 0
    
    for r in results:
        time_pos = r['time']
        # 已知的语音段: 0.5-1.5秒
        actual_voice = 1 if 0.5 <= time_pos <= 1.5 else 0
        if r['vad'] == actual_voice:
            correct_detections += 1
    
    accuracy = correct_detections / total_frames * 100
    print(f"基于已知信号的检测准确率: {accuracy:.1f}%")
    
    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)
    
    return results

def main():
    """主函数"""
    try:
        # 运行演示
        results = demo_signal_processing()
        
        # 提示用户运行完整系统
        print("\n提示: 运行 'python main.py' 启动完整的实时处理系统")
        print("系统将显示实时波形、能量、过零率和语音检测结果")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")

if __name__ == "__main__":
    main()