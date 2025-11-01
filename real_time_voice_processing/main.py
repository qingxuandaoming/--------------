#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from real_time_voice_processing.config import Config
from real_time_voice_processing.runtime.engine import AudioRuntime
from real_time_voice_processing.runtime.audio_source import FileAudioSource
from real_time_voice_processing.ui.visualization import VisualizationUI


def main():
    # 初始化日志并加载配置（环境变量优先，YAML 可选）
    Config.setup_logging()
    yaml_path = os.environ.get("RTP_CONFIG_YAML")
    if yaml_path:
        Config.load_from_yaml(yaml_path)
    Config.load_from_env(prefix="RTP_")

    # 可选：从文件读取作为音频源（适合集成测试/演示）
    input_file = os.environ.get("RTP_INPUT_FILE")
    runtime = AudioRuntime(audio_source=FileAudioSource(input_file, sample_rate=Config.SAMPLE_RATE) if input_file else None)
    ui = VisualizationUI(runtime)
    ui.run()


if __name__ == "__main__":
    main()