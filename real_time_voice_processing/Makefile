# 实时语音信号处理系统 - Makefile

# 环境变量（默认针对 Windows；Linux/macOS 请按需修改）
PYTHON = py -3.10
VENV = .venv
PIP = $(VENV)/Scripts/pip
PYTHON_EXEC = $(VENV)/Scripts/python

# 目标文件
MAIN_SCRIPT = main.py
TEST_SCRIPT = tests
DEMO_SCRIPT = demo.py

# 默认目标
all: help

# 帮助信息
help:
	@echo "实时语音信号处理系统 - 构建脚本"
	@echo "=============================="
	@echo "make venv      - 创建虚拟环境"
	@echo "make install   - 安装依赖包"
	@echo "make run       - 运行主程序"
	@echo "make test      - 运行测试程序"
	@echo "make demo      - 运行演示程序"
	@echo "make clean     - 清理项目文件"
	@echo "make doc       - 生成文档"
	@echo "make all       - 显示帮助信息"

# 创建虚拟环境
venv:
    @echo "创建虚拟环境 (Python 3.10)..."
    $(PYTHON) -m venv $(VENV)
    @echo "虚拟环境创建完成：请执行 .\\.venv\\Scripts\\Activate.ps1 激活"

# 安装依赖包
install: venv
    @echo "安装依赖包..."
    $(PIP) install --upgrade pip
    $(PIP) install -r requirements.txt
    @echo "依赖包安装完成"

# 运行主程序
run:
    @echo "运行主程序..."
    $(PYTHON_EXEC) $(MAIN_SCRIPT)

# 运行测试程序
test:
    @echo "运行测试程序 (pytest)..."
    $(PYTHON_EXEC) -m pytest -q

# 运行演示程序
demo:
    @echo "运行演示程序..."
    $(PYTHON_EXEC) $(DEMO_SCRIPT)

# 清理项目文件
clean:
    @echo "清理项目文件..."
    powershell -Command "Remove-Item -Recurse -Force __pycache__, $(VENV); Get-ChildItem -Include *.npz,*.wav,*.log -File | Remove-Item -Force"
    @echo "清理完成"

# 生成文档
doc:
	@echo "生成项目文档..."
	@echo "README.md - 项目说明文档"
	@echo "课程设计论文.md - 完整学术论文"
	@echo "PPT汇报.md - 汇报材料"
	@echo "文档生成完成"

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

.PHONY: all help venv install run test demo clean doc lint format