#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# 允许 autodoc 导入项目包
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'real_time_voice_processing'))

project = '实时语音信号处理系统'
author = 'Course Project'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
]

# 支持 Markdown 与 reStructuredText
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# MyST 配置，让 $...$ 与 $$...$$ 渲染数学
myst_enable_extensions = [
    'dollarmath',
    'deflist',
]