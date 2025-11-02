# 实时语音信号处理系统 - Makefile（跨平台）

# OS 检测与变量
ifeq ($(OS),Windows_NT)
  PYTHON ?= py -3.10
  VENV := .venv
  VENV_BIN := $(VENV)/Scripts
else
  PYTHON ?= python3.10
  VENV := .venv
  VENV_BIN := $(VENV)/bin
endif

PIP := $(VENV_BIN)/pip
PYTHON_EXEC := $(VENV_BIN)/python

# 路径与脚本
MAIN_SCRIPT := real_time_voice_processing/main.py
DEMO_SCRIPT := real_time_voice_processing/demo.py
TEST_DIR := tests
REQ_PATH := real_time_voice_processing/requirements.txt

.PHONY: help venv install run test demo clean doc lint format

# 帮助信息
help:
	@echo "实时语音信号处理系统 - 构建脚本"
	@echo "=============================="
	@echo "make venv      - 创建虚拟环境"
	@echo "make install   - 安装依赖包"
	@echo "make run       - 运行主程序"
	@echo "make test      - 运行测试程序"
	@echo "make demo      - 运行演示程序"
	@echo "make doc       - 生成文档"
	@echo "make lint      - 检查代码风格"
	@echo "make format    - 格式化代码"
	@echo "make clean     - 清理项目文件"

# 创建虚拟环境
venv:
	@echo "创建虚拟环境 (Python 3.10)..."
	$(PYTHON) -m venv $(VENV)
	@echo "激活：Windows .\\.venv\\Scripts\\Activate.ps1 | Linux/macOS source .venv/bin/activate"

# 安装依赖包
install: venv
	@echo "安装依赖包..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQ_PATH)
	@echo "依赖包安装完成"

# 运行主程序
run:
	@echo "运行主程序..."
	$(PYTHON_EXEC) $(MAIN_SCRIPT)

# 运行测试程序
test:
	@echo "运行测试程序 (pytest)..."
	$(PYTHON_EXEC) -m pytest -q $(TEST_DIR)

# 运行演示程序
demo:
	@echo "运行演示程序..."
	$(PYTHON_EXEC) $(DEMO_SCRIPT)

# 生成文档
doc:
	@echo "生成项目文档..."
	$(PYTHON_EXEC) -m pip install -r $(REQ_PATH)
	$(PYTHON_EXEC) -m sphinx -b html docs docs/_build/html

# 检查代码风格
lint:
	@echo "检查代码风格..."
	$(PIP) install flake8 black
	$(PYTHON_EXEC) -m flake8 .
	$(PYTHON_EXEC) -m black --check .

# 格式化代码
format:
	@echo "格式化代码..."
	$(PIP) install black
	$(PYTHON_EXEC) -m black .

# 清理项目文件（占位）
clean:
	@echo "清理项目文件..."
	@echo "清理完成"