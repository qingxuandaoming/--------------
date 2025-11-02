#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI 样式工具

提供默认配色方案与样式表构建函数。

Notes
-----
- 样式表基于 Qt Stylesheet，适配 pyqtgraph 及基本控件。
"""

from typing import Dict

# 默认配色方案
DEFAULT_PALETTE: Dict[str, str] = {
    "PRIMARY": "#0A4DAA",   # 深蓝
    "ACCENT": "#2EA3F2",    # 亮蓝
    "GREEN": "#5F7865",     # 灰绿
    "GOLD": "#B8842D",      # 金黄
    "BEIGE": "#EACCA8",     # 米色
    "DARK": "#212C28",      # 深灰绿
    "FG": "#E8EAED",        # 前景/文字
}

# 浅色主题配色（与默认色系保持一致，仅背景/文字反转）
LIGHT_PALETTE: Dict[str, str] = {
    "PRIMARY": "#0A4DAA",
    "ACCENT": "#2EA3F2",
    "GREEN": "#5F7865",
    "GOLD": "#B8842D",
    "BEIGE": "#EACCA8",
    "DARK": "#FAFAFA",      # 作为浅色背景
    "FG": "#202124",        # 深色文字
}


def build_stylesheet(palette: Dict[str, str]) -> str:
    """
    构建 Qt 样式表字符串。

    Parameters
    ----------
    palette : dict
        颜色字典，包含 `PRIMARY`、`ACCENT`、`GREEN`、`GOLD`、`BEIGE`、`DARK`、`FG` 键。

    Returns
    -------
    str
        Qt 样式表字符串。
    """
    p = palette
    return f"""
    QWidget {{ color: {p['FG']}; }}
    QGroupBox {{ border: 1px solid {p['GREEN']}; border-radius: 6px; margin-top: 6px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {p['ACCENT']}; }}
    QPushButton {{
        background-color: {p['PRIMARY']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 14px;
    }}
    QPushButton:hover {{ background-color: {p['ACCENT']}; }}
    QPushButton#stop {{ background-color: {p['GREEN']}; }}
    QPushButton#save {{ background-color: {p['GOLD']}; }}
    QLabel {{ color: {p['FG']}; }}
    QComboBox {{ background: {p['BEIGE']}; color: black; border-radius: 6px; padding: 4px; }}
    QComboBox:disabled {{ background: #C8C8C8; color: #666; }}
    /* 单选与复选的统一视觉 */
    QRadioButton, QCheckBox {{ color: {p['FG']}; }}
    QRadioButton::indicator, QCheckBox::indicator {{ width: 16px; height: 16px; }}
    QRadioButton::indicator:unchecked, QCheckBox::indicator:unchecked {{ border: 2px solid {p['ACCENT']}; background: transparent; border-radius: 8px; }}
    QRadioButton::indicator:checked, QCheckBox::indicator:checked {{ background: {p['ACCENT']}; border: 2px solid {p['ACCENT']}; border-radius: 8px; }}
    QRadioButton:disabled, QCheckBox:disabled {{ color: #8A8F99; }}
    /* 提示框 */
    QToolTip {{ color: black; background-color: {p['BEIGE']}; border: 1px solid {p['GOLD']}; padding: 4px; }}
    """