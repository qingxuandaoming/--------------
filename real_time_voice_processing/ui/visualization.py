#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class VisualizationUI:
    """可视化界面：负责图形界面与交互"""

    def __init__(self, runtime, title="实时语音信号处理系统"):
        self.runtime = runtime
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(1200, 800)

        self._init_ui()
        self._init_timer()

    def _init_ui(self):
        self.waveform_plot = self.win.addPlot(title="实时音频波形", row=0, col=0)
        self.waveform_curve = self.waveform_plot.plot(pen="b")
        self.waveform_plot.setYRange(-32768, 32768)
        self.waveform_plot.setXRange(0, 4096)
        self.waveform_plot.setLabel("left", "幅度")
        self.waveform_plot.setLabel("bottom", "样本点")

        self.energy_plot = self.win.addPlot(title="短时能量", row=1, col=0)
        self.energy_curve = self.energy_plot.plot(pen="r")
        self.energy_plot.setYRange(0, 1e10)
        self.energy_plot.setXRange(0, 100)
        self.energy_plot.setLabel("left", "能量")
        self.energy_plot.setLabel("bottom", "帧数")

        self.zcr_plot = self.win.addPlot(title="过零率", row=2, col=0)
        self.zcr_curve = self.zcr_plot.plot(pen="g")
        self.zcr_plot.setYRange(0, 0.5)
        self.zcr_plot.setXRange(0, 100)
        self.zcr_plot.setLabel("left", "过零率")
        self.zcr_plot.setLabel("bottom", "帧数")

        self.vad_plot = self.win.addPlot(title="语音活动检测", row=3, col=0)
        self.vad_curve = self.vad_plot.plot(pen="m", fillLevel=0, brush=(100, 100, 255, 50))
        self.vad_plot.setYRange(-0.1, 1.1)
        self.vad_plot.setXRange(0, 100)
        self.vad_plot.setLabel("left", "语音存在")
        self.vad_plot.setLabel("bottom", "帧数")

        self.ctrl_layout = QtGui.QHBoxLayout()
        self.start_btn = QtGui.QPushButton("开始处理")
        self.stop_btn = QtGui.QPushButton("停止处理")
        self.save_btn = QtGui.QPushButton("保存数据")

        self.start_btn.clicked.connect(self.runtime.start)
        self.stop_btn.clicked.connect(self.runtime.stop)
        self.save_btn.clicked.connect(lambda: print(f"数据已保存到: {self.runtime.save_data()}"))

        self.ctrl_layout.addWidget(self.start_btn)
        self.ctrl_layout.addWidget(self.stop_btn)
        self.ctrl_layout.addWidget(self.save_btn)

        self.ctrl_widget = QtGui.QWidget()
        self.ctrl_widget.setLayout(self.ctrl_layout)
        self.win.addWidget(self.ctrl_widget, row=4, col=0)

    def _init_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(50)

    def _update_plots(self):
        recent_audio = self.runtime.get_recent_audio()
        if len(recent_audio) > 0:
            self.waveform_curve.setData(recent_audio)

        energies, zcrs, vads = self.runtime.get_recent_processed()
        if len(energies) > 0:
            x_data = np.arange(len(energies))
            self.energy_curve.setData(x_data, energies)
            self.zcr_curve.setData(x_data, zcrs)
            self.vad_curve.setData(x_data, vads)

    def run(self):
        sys.exit(self.app.exec_())