#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试文件
用于验证各个模块的功能正确性
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_processing import SignalProcessing
from config import Config

class SystemTester:
    """系统测试类"""
    
    def __init__(self):
        """初始化测试器"""
        print("=" * 60)
        print("实时语音信号处理系统 - 功能测试")
        print("=" * 60)
        
    def test_window_functions(self):
        """测试窗函数生成"""
        print("\n1. 测试窗函数生成...")
        
        frame_size = 320
        hamming = SignalProcessing.hamming_window(frame_size)
        hanning = SignalProcessing.hanning_window(frame_size)
        rectangular = SignalProcessing.rectangular_window(frame_size)
        
        print(f"   汉明窗 - 长度: {len(hamming)}, 最大值: {np.max(hamming):.6f}, 最小值: {np.min(hamming):.6f}")
        print(f"   汉宁窗 - 长度: {len(hanning)}, 最大值: {np.max(hanning):.6f}, 最小值: {np.min(hanning):.6f}")
        print(f"   矩形窗 - 长度: {len(rectangular)}, 最大值: {np.max(rectangular):.6f}, 最小值: {np.min(rectangular):.6f}")
        
        # 验证窗函数的基本特性
        assert len(hamming) == frame_size, "汉明窗长度错误"
        assert len(hanning) == frame_size, "汉宁窗长度错误"
        assert len(rectangular) == frame_size, "矩形窗长度错误"
        assert abs(np.max(hamming) - 1.0) < 1e-4, f"汉明窗最大值应为1.0，实际为{np.max(hamming)}"
        assert abs(np.max(hanning) - 1.0) < 1e-4, f"汉宁窗最大值应为1.0，实际为{np.max(hanning)}"
        assert np.all(rectangular == 1.0), "矩形窗应为全1"
        
        print("   ✓ 窗函数测试通过")
    
    def test_short_time_energy(self):
        """测试短时能量计算"""
        print("\n2. 测试短时能量计算...")
        
        # 生成测试信号
        frame_size = 320
        test_signal = np.random.randn(frame_size) * 1000  # 随机噪声
        silence_signal = np.zeros(frame_size)  # 静音
        
        # 计算能量
        energy_test = SignalProcessing.calculate_short_time_energy(test_signal)
        energy_silence = SignalProcessing.calculate_short_time_energy(silence_signal)
        
        print(f"   测试信号能量: {energy_test:.2e}")
        print(f"   静音信号能量: {energy_silence:.2e}")
        
        # 验证能量计算
        assert energy_test > 0, "测试信号能量应为正值"
        assert np.isclose(energy_silence, 0), "静音信号能量应为0"
        
        print("   ✓ 短时能量测试通过")
    
    def test_zero_crossing_rate(self):
        """测试过零率计算"""
        print("\n3. 测试过零率计算...")
        
        # 生成测试信号
        frame_size = 320
        freq = 100  # 100Hz正弦波
        t = np.arange(frame_size) / Config.SAMPLE_RATE
        sine_wave = np.sin(2 * np.pi * freq * t) * 1000
        
        # 计算过零率
        zcr_sine = SignalProcessing.calculate_zero_crossing_rate(sine_wave)
        zcr_silence = SignalProcessing.calculate_zero_crossing_rate(np.zeros(frame_size))
        
        print(f"   正弦波过零率: {zcr_sine:.4f}")
        print(f"   静音信号过零率: {zcr_silence:.4f}")
        
        # 理论过零率计算 (每个周期过零2次)
        periods_per_frame = (freq * frame_size) / Config.SAMPLE_RATE
        theoretical_zcr = (periods_per_frame * 2) / frame_size
        
        print(f"   理论过零率: {theoretical_zcr:.4f}")
        
        # 验证过零率计算
        assert abs(zcr_sine - theoretical_zcr) < 0.01, "正弦波过零率计算错误"
        assert np.isclose(zcr_silence, 0), "静音信号过零率应为0"
        
        print("   ✓ 过零率测试通过")
    
    def test_autocorrelation(self):
        """测试自相关计算"""
        print("\n4. 测试自相关计算...")
        
        # 生成测试信号
        frame_size = 320
        freq = 100
        t = np.arange(frame_size) / Config.SAMPLE_RATE
        sine_wave = np.sin(2 * np.pi * freq * t)
        
        # 计算自相关
        acf = SignalProcessing.calculate_short_time_autocorrelation(sine_wave, max_lag=100)
        
        print(f"   自相关函数长度: {len(acf)}")
        print(f"   自相关最大值: {np.max(acf):.4f}")
        print(f"   自相关最小值: {np.min(acf):.4f}")
        
        # 验证自相关特性
        assert np.isclose(acf[0], 1.0), "自相关在延迟0处应为1"
        assert len(acf) == 100, "自相关长度错误"
        
        print("   ✓ 自相关测试通过")
    
    def test_voice_activity_detection(self):
        """测试语音活动检测"""
        print("\n5. 测试语音活动检测...")
        
        # 测试不同情况
        test_cases = [
            (10000, 0.2, "强语音"),  # 高能量高过零率
            (500, 0.2, "低能量高过零率"),  # 低能量高过零率
            (10000, 0.05, "高能量低过零率"),  # 高能量低过零率
            (500, 0.05, "静音")  # 低能量低过零率
        ]
        
        for energy, zcr, case in test_cases:
            result = SignalProcessing.voice_activity_detection(energy, zcr)
            print(f"   {case}: 能量={energy}, 过零率={zcr:.3f} → {'语音' if result else '静音'}")
        
        # 验证检测逻辑
        assert SignalProcessing.voice_activity_detection(10000, 0.2) == 1, "强语音应被检测为语音"
        assert SignalProcessing.voice_activity_detection(500, 0.05) == 0, "静音应被检测为静音"
        
        print("   ✓ 语音活动检测测试通过")
    
    def test_framing(self):
        """测试分帧功能"""
        print("\n6. 测试分帧功能...")
        
        # 生成测试信号
        signal_length = 1000
        signal = np.random.randn(signal_length)
        
        # 分帧
        frames = SignalProcessing.framing(signal, Config.FRAME_SIZE, Config.HOP_SIZE)
        
        print(f"   原始信号长度: {signal_length}")
        print(f"   帧大小: {Config.FRAME_SIZE}")
        print(f"   帧移: {Config.HOP_SIZE}")
        print(f"   帧数: {len(frames)}")
        
        # 验证分帧结果
        expected_frames = 1 + int(np.ceil((signal_length - Config.FRAME_SIZE) / Config.HOP_SIZE))
        assert len(frames) == expected_frames, "帧数计算错误"
        assert frames.shape[1] == Config.FRAME_SIZE, "帧大小错误"
        
        print("   ✓ 分帧测试通过")
    
    def run_all_tests(self):
        """运行所有测试"""
        self.test_window_functions()
        self.test_short_time_energy()
        self.test_zero_crossing_rate()
        self.test_autocorrelation()
        self.test_voice_activity_detection()
        self.test_framing()
        
        print("\n" + "=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)

if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all_tests()