#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
from collections import deque
import numpy as np
import pyaudio

from real_time_voice_processing.config import Config
from real_time_voice_processing.signal_processing import SignalProcessing


class AudioRuntime:
    """运行时引擎：负责音频采集与信号处理线程"""

    def __init__(self):
        # 基本参数
        self.format = Config.AUDIO_FORMAT
        self.channels = Config.CHANNELS
        self.rate = Config.SAMPLE_RATE
        self.chunk = Config.CHUNK_SIZE
        self.frame_size = Config.FRAME_SIZE
        self.hop_size = Config.HOP_SIZE

        # 窗函数
        self.window = SignalProcessing.hamming_window(self.frame_size)

        # 阈值
        self.energy_threshold = Config.ENERGY_THRESHOLD
        self.zcr_threshold = Config.ZCR_THRESHOLD

        # 缓冲区
        self.audio_buffer = deque(maxlen=Config.AUDIO_BUFFER_SIZE)
        self.processed_data = deque(maxlen=Config.PROCESSED_DATA_BUFFER_SIZE)
        # 特征历史（用于自适应VAD）
        self.energy_history = deque(maxlen=256)
        self.zcr_history = deque(maxlen=256)

        # 线程控制
        self.is_running = False
        self.audio_thread = None
        self.processing_thread = None
        self.lock = threading.Lock()

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.audio_thread = threading.Thread(target=self._audio_capture_thread, daemon=True)
            self.processing_thread = threading.Thread(target=self._signal_processing_thread, daemon=True)
            self.audio_thread.start()
            self.processing_thread.start()

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.audio_thread:
                self.audio_thread.join()
            if self.processing_thread:
                self.processing_thread.join()

    def _audio_capture_thread(self):
        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
            while self.is_running:
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                with self.lock:
                    self.audio_buffer.append(audio_data)
        except Exception:
            pass
        finally:
            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            finally:
                p.terminate()

    def _signal_processing_thread(self):
        overlap_buffer = np.array([], dtype=np.int16)
        sleep_time = Config.THREAD_SLEEP_TIME

        while self.is_running:
            if len(self.audio_buffer) == 0:
                time.sleep(sleep_time)
                continue
            with self.lock:
                audio_data = self.audio_buffer.popleft()
            overlap_buffer = np.concatenate((overlap_buffer, audio_data))

            while len(overlap_buffer) >= self.frame_size:
                frame = overlap_buffer[: self.frame_size].astype(np.float32)
                overlap_buffer = overlap_buffer[self.hop_size :]

                windowed_frame = frame * self.window
                energy = SignalProcessing.calculate_short_time_energy(windowed_frame)
                zcr = SignalProcessing.calculate_zero_crossing_rate(windowed_frame)
                vad = SignalProcessing.voice_activity_detection(
                    energy, zcr, self.energy_threshold, self.zcr_threshold
                )

                # 频域特征与自适应VAD
                spec_entropy = SignalProcessing.calculate_spectral_entropy(
                    windowed_frame, n_fft=Config.SPECTRAL_ENTROPY_N_FFT
                )
                vad_adaptive = SignalProcessing.adaptive_voice_activity_detection(
                    energy,
                    zcr,
                    list(self.energy_history),
                    list(self.zcr_history),
                    energy_k=Config.ADAPTIVE_VAD_ENERGY_K,
                    zcr_k=Config.ADAPTIVE_VAD_ZCR_K,
                    min_history=Config.ADAPTIVE_VAD_HISTORY_MIN,
                    fallback_energy_threshold=self.energy_threshold,
                    fallback_zcr_threshold=self.zcr_threshold,
                )
                mfcc = SignalProcessing.compute_mfcc(
                    windowed_frame,
                    sample_rate=self.rate,
                    num_ceps=Config.NUM_MFCC,
                    n_fft=Config.MFCC_N_FFT,
                    n_filters=Config.MEL_FILTERS,
                    lifter=Config.MFCC_LIFTER,
                    pre_emphasis=None,
                )

                with self.lock:
                    self.energy_history.append(float(energy))
                    self.zcr_history.append(float(zcr))
                    self.processed_data.append(
                        {
                            "energy": float(energy),
                            "zcr": float(zcr),
                            "vad": int(vad),
                            "spec_entropy": float(spec_entropy),
                            "vad_adaptive": int(vad_adaptive),
                            "mfcc": mfcc.tolist(),
                        }
                    )

    def get_recent_audio(self):
        with self.lock:
            if len(self.audio_buffer) == 0:
                return np.array([], dtype=np.int16)
            recent_audio = np.concatenate(list(self.audio_buffer))
        length = Config.WAVEFORM_DISPLAY_LENGTH
        if len(recent_audio) > length:
            recent_audio = recent_audio[-length:]
        return recent_audio

    def get_recent_processed(self, max_display=None):
        if max_display is None:
            max_display = Config.MAX_DISPLAY_FRAMES
        with self.lock:
            if len(self.processed_data) == 0:
                return np.array([]), np.array([]), np.array([])
            energies = [d["energy"] for d in self.processed_data]
            zcrs = [d["zcr"] for d in self.processed_data]
            vads = [d["vad"] for d in self.processed_data]
        if len(energies) > max_display:
            energies = energies[-max_display:]
            zcrs = zcrs[-max_display:]
            vads = vads[-max_display:]
        return np.array(energies), np.array(zcrs), np.array(vads)

    def save_data(self, directory=None):
        if directory is None:
            directory = Config.SAVE_DIRECTORY
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/voice_processing_data_{timestamp}.npz"
        energies, zcrs, vads = self.get_recent_processed(max_display=Config.PROCESSED_DATA_BUFFER_SIZE)
        # 其他可选特征
        with self.lock:
            spec_entropies = [d.get("spec_entropy", np.nan) for d in self.processed_data]
            vads_adaptive = [d.get("vad_adaptive", np.nan) for d in self.processed_data]
        if len(spec_entropies) > Config.PROCESSED_DATA_BUFFER_SIZE:
            spec_entropies = spec_entropies[-Config.PROCESSED_DATA_BUFFER_SIZE:]
            vads_adaptive = vads_adaptive[-Config.PROCESSED_DATA_BUFFER_SIZE:]
        np.savez(
            filename,
            energies=np.array(energies),
            zcrs=np.array(zcrs),
            vads=np.array(vads),
            spec_entropy=np.array(spec_entropies, dtype=np.float32),
            vads_adaptive=np.array(vads_adaptive, dtype=np.float32),
            sample_rate=self.rate,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
        )
        return filename