#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from real_time_voice_processing.runtime.engine import AudioRuntime
from real_time_voice_processing.ui.visualization import VisualizationUI


def main():
    runtime = AudioRuntime()
    ui = VisualizationUI(runtime)
    ui.run()


if __name__ == "__main__":
    main()