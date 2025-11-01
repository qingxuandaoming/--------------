#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号处理算法模块
包含语音信号处理的核心算法实现
"""

import numpy as np
from scipy.signal import correlate
from scipy.fftpack import dct

class SignalProcessing:
    """信号处理算法类"""
    
    @staticmethod
    def hamming_window(length):
        """生成汉明窗"""
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(length) / (length - 1))
    
    @staticmethod
    def hanning_window(length):
        """生成汉宁窗"""
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))
    
    @staticmethod
    def rectangular_window(length):
        """生成矩形窗"""
        return np.ones(length)
    
    @staticmethod
    def calculate_short_time_energy(frame, window_type='hamming'):
        """
        计算短时能量
        
        参数:
        frame: 输入音频帧
        window_type: 窗函数类型 ('hamming', 'hanning', 'rectangular')
        
        返回:
        energy: 短时能量值
        """
        # 选择窗函数
        if window_type == 'hamming':
            window = SignalProcessing.hamming_window(len(frame))
        elif window_type == 'hanning':
            window = SignalProcessing.hanning_window(len(frame))
        else:  # rectangular
            window = SignalProcessing.rectangular_window(len(frame))
        
        # 加窗
        windowed_frame = frame * window
        
        # 计算能量
        energy = np.sum(windowed_frame ** 2)
        
        return energy
    
    @staticmethod
    def calculate_zero_crossing_rate(frame):
        """
        计算过零率
        
        参数:
        frame: 输入音频帧
        
        返回:
        zcr: 过零率值
        """
        # 去除直流分量
        frame = frame - np.mean(frame)
        
        # 计算过零次数
        crossings = np.where(np.diff(np.signbit(frame)))[0]
        zcr = len(crossings) / len(frame)
        
        return zcr
    
    @staticmethod
    def calculate_short_time_autocorrelation(frame, max_lag=None):
        """
        计算短时自相关函数
        
        参数:
        frame: 输入音频帧
        max_lag: 最大延迟
        
        返回:
        acf: 自相关函数值
        """
        if max_lag is None:
            max_lag = len(frame)
        
        # 计算自相关
        acf = correlate(frame, frame, mode='full')
        acf = acf[-len(frame):][:max_lag]
        
        # 归一化
        acf = acf / np.max(acf)
        
        return acf
    
    @staticmethod
    def calculate_average_magnitude_difference(frame, max_lag=None):
        """
        计算平均幅度差函数
        
        参数:
        frame: 输入音频帧
        max_lag: 最大延迟
        
        返回:
        amdf: 平均幅度差函数值
        """
        if max_lag is None:
            max_lag = len(frame)
        
        amdf = []
        for l in range(1, max_lag):
            diff = np.mean(np.abs(frame[l:] - frame[:-l]))
            amdf.append(diff)
        
        return np.array(amdf)
    
    @staticmethod
    def voice_activity_detection(energy, zcr, energy_threshold=1000, zcr_threshold=0.1):
        """
        语音活动检测
        
        参数:
        energy: 短时能量
        zcr: 过零率
        energy_threshold: 能量阈值
        zcr_threshold: 过零率阈值
        
        返回:
        vad_result: 语音活动检测结果 (1=语音, 0=静音)
        """
        # 双门限检测算法
        if energy > energy_threshold and zcr > zcr_threshold:
            return 1
        return 0
    
    @staticmethod
    def preemphasis(signal, alpha=0.97):
        """
        预加重处理
        
        参数:
        signal: 输入信号
        alpha: 预加重系数
        
        返回:
        preemphasized_signal: 预加重后的信号
        """
        return np.append(signal[0], signal[1:] - alpha * signal[:-1])
    
    @staticmethod
    def framing(signal, frame_size, hop_size, window_type='hamming'):
        """
        信号分帧
        
        参数:
        signal: 输入信号
        frame_size: 帧大小
        hop_size: 帧移
        window_type: 窗函数类型
        
        返回:
        frames: 分帧后的信号
        """
        signal_length = len(signal)
        num_frames = 1 + int(np.ceil((signal_length - frame_size) / hop_size))
        
        # 补零以确保所有帧都有完整的长度
        pad_length = (num_frames - 1) * hop_size + frame_size
        padded_signal = np.pad(signal, (0, pad_length - signal_length), mode='constant')
        
        # 生成帧索引
        indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * hop_size, hop_size), (frame_size, 1)).T
        
        # 提取帧
        frames = padded_signal[indices.astype(np.int32, copy=False)]
        
        # 选择窗函数
        if window_type == 'hamming':
            window = SignalProcessing.hamming_window(frame_size)
        elif window_type == 'hanning':
            window = SignalProcessing.hanning_window(frame_size)
        else:
            window = SignalProcessing.rectangular_window(frame_size)
        
        # 加窗
        frames = frames * window
        
        return frames

    # =========================
    # 频域特征与高级检测算法
    # =========================

    @staticmethod
    def _hz_to_mel(f_hz):
        """Hz 转 Mel"""
        return 2595.0 * np.log10(1.0 + (f_hz / 700.0))

    @staticmethod
    def _mel_to_hz(m_mel):
        """Mel 转 Hz"""
        return 700.0 * (10.0 ** (m_mel / 2595.0) - 1.0)

    @staticmethod
    def mel_filterbank(sample_rate, n_fft, n_filters=26, fmin=0.0, fmax=None):
        """
        生成 Mel 滤波器组（三角形滤波器）

        参数:
        - sample_rate: 采样率
        - n_fft: FFT 点数
        - n_filters: 滤波器数量
        - fmin: 最低频率（Hz）
        - fmax: 最高频率（Hz），默认 Nyquist

        返回:
        - fbank: 形状为 (n_filters, n_fft//2 + 1) 的滤波器组矩阵
        """
        if fmax is None:
            fmax = sample_rate / 2.0

        # Mel 均匀分布的锚点
        low_mel = SignalProcessing._hz_to_mel(fmin)
        high_mel = SignalProcessing._hz_to_mel(fmax)
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points = SignalProcessing._mel_to_hz(mel_points)

        # 对应到 FFT 频率 bin
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        bins = np.clip(bins, 0, n_fft // 2)

        fbank = np.zeros((n_filters, n_fft // 2 + 1))
        for m in range(1, n_filters + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]

            if f_m_minus == f_m:
                f_m_minus = max(0, f_m - 1)
            if f_m == f_m_plus:
                f_m_plus = min(n_fft // 2, f_m + 1)

            # 上升段
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            # 下降段
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return fbank

    @staticmethod
    def compute_mfcc(frame, sample_rate, num_ceps=13, n_fft=512, n_filters=26, lifter=22, pre_emphasis=None):
        """
        计算单帧 MFCC（Mel-Frequency Cepstral Coefficients）

        参数:
        - frame: 单帧信号（已加窗）
        - sample_rate: 采样率
        - num_ceps: 倒谱系数数量（常用 12-13）
        - n_fft: FFT 点数
        - n_filters: Mel 滤波器数量（常用 20-40）
        - lifter: 倒谱升力系数（提升高阶系数的权重）
        - pre_emphasis: 预加重系数（可选），如果提供则对帧做预加重

        返回:
        - mfcc: 形状为 (num_ceps,) 的 MFCC 向量
        """
        # 预加重（可选）
        if pre_emphasis is not None:
            frame = SignalProcessing.preemphasis(frame, alpha=float(pre_emphasis))

        # 功率谱（半谱）
        power_spec = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2

        # Mel 滤波器组与滤波能量
        fbank = SignalProcessing.mel_filterbank(sample_rate, n_fft, n_filters)
        filter_energies = np.dot(power_spec, fbank.T)
        filter_energies = np.where(filter_energies == 0.0, np.finfo(float).eps, filter_energies)

        # 对数与 DCT-II
        log_energies = np.log(filter_energies)
        mfcc = dct(log_energies, type=2, norm='ortho')[:num_ceps]

        # 倒谱升力（提升高阶倒谱系数）
        if lifter is not None and lifter > 0:
            n = np.arange(num_ceps)
            lift = 1 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
            mfcc = mfcc * lift

        return mfcc.astype(np.float32)

    @staticmethod
    def calculate_spectral_entropy(frame, n_fft=512, eps=1e-12):
        """
        计算谱熵（Spectral Entropy），反映频谱分布的离散程度。

        参数:
        - frame: 单帧信号（已加窗）
        - n_fft: FFT 点数
        - eps: 避免 log(0) 的极小量

        返回:
        - entropy: 归一化的谱熵值，范围约 [0, 1]
        """
        power_spec = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
        total_power = np.sum(power_spec) + eps
        p = power_spec / total_power
        entropy = -np.sum(p * np.log2(p + eps))
        # 归一化到 [0, 1]
        entropy /= np.log2(p.size)
        return float(entropy)

    @staticmethod
    def adaptive_voice_activity_detection(energy, zcr, energy_hist, zcr_hist, energy_k=3.0, zcr_k=1.0, min_history=20, fallback_energy_threshold=1000, fallback_zcr_threshold=0.1):
        """
        自适应语音活动检测（Adaptive VAD）
        使用历史能量/ZCR 的稳健基线（median + k * MAD）动态估计阈值。

        参数:
        - energy, zcr: 当前帧的能量与过零率
        - energy_hist, zcr_hist: 历史能量与过零率序列（list/ndarray）
        - energy_k, zcr_k: MAD 的缩放系数（越大越保守）
        - min_history: 启用自适应判决所需的最小历史长度
        - fallback_energy_threshold, fallback_zcr_threshold: 历史不足时使用的回退阈值

        返回:
        - vad_result: 1 表示语音，0 表示静音
        """
        try:
            e_hist = np.asarray(energy_hist, dtype=np.float32)
            z_hist = np.asarray(zcr_hist, dtype=np.float32)
        except Exception:
            e_hist = np.array([], dtype=np.float32)
            z_hist = np.array([], dtype=np.float32)

        if e_hist.size < min_history or z_hist.size < min_history:
            return 1 if (energy > fallback_energy_threshold and zcr > fallback_zcr_threshold) else 0

        # 基线估计（Median + k * MAD）
        e_med = np.median(e_hist)
        e_mad = np.median(np.abs(e_hist - e_med)) + 1e-8
        z_med = np.median(z_hist)
        z_mad = np.median(np.abs(z_hist - z_med)) + 1e-8

        e_thr = e_med + energy_k * e_mad
        z_thr = z_med + zcr_k * z_mad

        return 1 if (energy > e_thr and zcr > z_thr) else 0
    