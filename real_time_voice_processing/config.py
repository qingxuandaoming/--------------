#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置文件
包含所有可配置的参数
"""

class Config:
    """系统配置类"""
    
    # 音频参数
    AUDIO_FORMAT = 1  # pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 16000  # 16kHz 语音处理标准采样率
    CHUNK_SIZE = 1024  # 每次读取的数据块大小
    FRAME_DURATION = 20  # 帧长（毫秒）
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # 320个样本点
    HOP_SIZE = FRAME_SIZE // 2  # 帧移，50%重叠
    
    # 信号处理参数
    WINDOW_TYPE = 'hamming'  # 窗函数类型
    PREEMPHASIS_ALPHA = 0.97  # 预加重系数
    
    # 频域与特征参数
    NUM_MFCC = 13  # MFCC 倒谱系数数量
    MFCC_N_FFT = 512  # MFCC 计算的 FFT 点数
    MEL_FILTERS = 26  # Mel 滤波器数量
    MFCC_LIFTER = 22  # MFCC 倒谱升力系数
    SPECTRAL_ENTROPY_N_FFT = 512  # 谱熵计算的 FFT 点数
    
    # 语音活动检测参数
    ENERGY_THRESHOLD = 1000  # 能量阈值
    ZCR_THRESHOLD = 0.1  # 过零率阈值
    
    # 自适应 VAD 参数
    ADAPTIVE_VAD_HISTORY_MIN = 20
    ADAPTIVE_VAD_ENERGY_K = 3.0
    ADAPTIVE_VAD_ZCR_K = 1.0
    
    # 可视化参数
    PLOT_UPDATE_INTERVAL = 50  # 界面更新间隔（毫秒）
    MAX_DISPLAY_FRAMES = 100  # 最大显示帧数
    WAVEFORM_DISPLAY_LENGTH = 4096  # 波形图显示长度
    
    # 缓冲区参数
    AUDIO_BUFFER_SIZE = 4  # 音频缓冲区大小
    PROCESSED_DATA_BUFFER_SIZE = 100  # 处理数据缓冲区大小
    
    # 线程参数
    THREAD_SLEEP_TIME = 0.001  # 线程休眠时间（秒）
    
    # 文件保存参数
    SAVE_DIRECTORY = '.'  # 保存目录
    
    @staticmethod
    def print_config():
        """打印配置信息"""
        print("=" * 50)
        print("实时语音信号处理系统 - 配置信息")
        print("=" * 50)
        print(f"音频格式: {Config.AUDIO_FORMAT} (paInt16)")
        print(f"声道数: {Config.CHANNELS}")
        print(f"采样率: {Config.SAMPLE_RATE} Hz")
        print(f"块大小: {Config.CHUNK_SIZE}")
        print(f"帧长: {Config.FRAME_SIZE} 样本点 ({Config.FRAME_DURATION}ms)")
        print(f"帧移: {Config.HOP_SIZE} 样本点")
        print(f"窗函数: {Config.WINDOW_TYPE}")
        print(f"预加重系数: {Config.PREEMPHASIS_ALPHA}")
        print(f"MFCC: num={Config.NUM_MFCC}, n_fft={Config.MFCC_N_FFT}, mel_filters={Config.MEL_FILTERS}, lifter={Config.MFCC_LIFTER}")
        print(f"谱熵 FFT 点数: {Config.SPECTRAL_ENTROPY_N_FFT}")
        print(f"能量阈值: {Config.ENERGY_THRESHOLD}")
        print(f"过零率阈值: {Config.ZCR_THRESHOLD}")
        print(f"自适应VAD: history_min={Config.ADAPTIVE_VAD_HISTORY_MIN}, energy_k={Config.ADAPTIVE_VAD_ENERGY_K}, zcr_k={Config.ADAPTIVE_VAD_ZCR_K}")
        print("=" * 50)

# 测试配置
if __name__ == "__main__":
    Config.print_config()