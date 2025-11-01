#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from real_time_voice_processing.signal_processing import SignalProcessing
from real_time_voice_processing.config import Config


def test_window_functions():
    frame_size = 320
    hamming = SignalProcessing.hamming_window(frame_size)
    hanning = SignalProcessing.hanning_window(frame_size)
    rectangular = SignalProcessing.rectangular_window(frame_size)

    assert len(hamming) == frame_size
    assert len(hanning) == frame_size
    assert len(rectangular) == frame_size
    assert abs(np.max(hamming) - 1.0) < 1e-4
    assert abs(np.max(hanning) - 1.0) < 1e-4
    assert np.all(rectangular == 1.0)


def test_short_time_energy():
    frame_size = 320
    test_signal = np.random.randn(frame_size) * 1000
    silence_signal = np.zeros(frame_size)

    energy_test = SignalProcessing.calculate_short_time_energy(test_signal)
    energy_silence = SignalProcessing.calculate_short_time_energy(silence_signal)

    assert energy_test > 0
    assert np.isclose(energy_silence, 0)


def test_zero_crossing_rate():
    frame_size = 320
    freq = 100
    t = np.arange(frame_size) / Config.SAMPLE_RATE
    sine_wave = np.sin(2 * np.pi * freq * t) * 1000

    zcr_sine = SignalProcessing.calculate_zero_crossing_rate(sine_wave)
    zcr_silence = SignalProcessing.calculate_zero_crossing_rate(np.zeros(frame_size))

    periods_per_frame = (freq * frame_size) / Config.SAMPLE_RATE
    theoretical_zcr = (periods_per_frame * 2) / frame_size

    assert abs(zcr_sine - theoretical_zcr) < 0.01
    assert np.isclose(zcr_silence, 0)


def test_autocorrelation():
    frame_size = 320
    freq = 100
    t = np.arange(frame_size) / Config.SAMPLE_RATE
    sine_wave = np.sin(2 * np.pi * freq * t)

    max_lag = 100
    acf = SignalProcessing.calculate_short_time_autocorrelation(sine_wave, max_lag=max_lag)

    assert np.isclose(acf[0], 1.0)
    assert len(acf) == max_lag


def test_voice_activity_detection():
    assert SignalProcessing.voice_activity_detection(10000, 0.2) == 1
    assert SignalProcessing.voice_activity_detection(500, 0.05) == 0


def test_framing():
    signal_length = 1000
    signal = np.random.randn(signal_length)
    frames = SignalProcessing.framing(signal, Config.FRAME_SIZE, Config.HOP_SIZE)

    expected_frames = 1 + int(np.ceil((signal_length - Config.FRAME_SIZE) / Config.HOP_SIZE))
    assert len(frames) == expected_frames
    assert frames.shape[1] == Config.FRAME_SIZE


def test_spectral_entropy_and_mfcc():
    frame_size = Config.FRAME_SIZE
    t = np.arange(frame_size) / Config.SAMPLE_RATE
    # 纯音与白噪声帧
    sine_wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = np.random.randn(frame_size).astype(np.float32)

    # 加窗
    sine_wave *= SignalProcessing.hamming_window(frame_size)
    noise *= SignalProcessing.hamming_window(frame_size)

    ent_tone = SignalProcessing.calculate_spectral_entropy(sine_wave, n_fft=Config.SPECTRAL_ENTROPY_N_FFT)
    ent_noise = SignalProcessing.calculate_spectral_entropy(noise, n_fft=Config.SPECTRAL_ENTROPY_N_FFT)
    assert 0.0 <= ent_tone <= 1.0
    assert 0.0 <= ent_noise <= 1.0
    assert ent_noise > ent_tone  # 噪声的谱更均匀，熵更高

    mfcc = SignalProcessing.compute_mfcc(
        sine_wave,
        sample_rate=Config.SAMPLE_RATE,
        num_ceps=Config.NUM_MFCC,
        n_fft=Config.MFCC_N_FFT,
        n_filters=Config.MEL_FILTERS,
        lifter=Config.MFCC_LIFTER,
    )
    assert mfcc.shape == (Config.NUM_MFCC,)
    assert np.all(np.isfinite(mfcc))
    assert np.any(np.abs(mfcc) > 1e-6)


def test_adaptive_vad():
    # 历史为低能量、低ZCR（环境噪声/静音）
    energy_hist = np.random.uniform(100.0, 300.0, size=50)
    zcr_hist = np.random.uniform(0.01, 0.05, size=50)

    # 当前帧明显语音特征
    energy_cur = 5000.0
    zcr_cur = 0.2
    vad1 = SignalProcessing.adaptive_voice_activity_detection(
        energy_cur,
        zcr_cur,
        energy_hist,
        zcr_hist,
        energy_k=Config.ADAPTIVE_VAD_ENERGY_K,
        zcr_k=Config.ADAPTIVE_VAD_ZCR_K,
        min_history=Config.ADAPTIVE_VAD_HISTORY_MIN,
        fallback_energy_threshold=Config.ENERGY_THRESHOLD,
        fallback_zcr_threshold=Config.ZCR_THRESHOLD,
    )
    assert vad1 == 1

    # 当前帧低能量、低ZCR，应判为静音
    energy_cur2 = 200.0
    zcr_cur2 = 0.03
    vad2 = SignalProcessing.adaptive_voice_activity_detection(
        energy_cur2,
        zcr_cur2,
        energy_hist,
        zcr_hist,
        energy_k=Config.ADAPTIVE_VAD_ENERGY_K,
        zcr_k=Config.ADAPTIVE_VAD_ZCR_K,
        min_history=Config.ADAPTIVE_VAD_HISTORY_MIN,
        fallback_energy_threshold=Config.ENERGY_THRESHOLD,
        fallback_zcr_threshold=Config.ZCR_THRESHOLD,
    )
    assert vad2 == 0