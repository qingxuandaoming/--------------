# 用于 Windows PowerShell 的虚拟环境快速创建脚本（Python 3.10）
Param(
    [string]$PythonLauncher = "py -3.10"
)

Write-Host "使用 Python 启动器: $PythonLauncher"

try {
    & $PythonLauncher -m venv .venv
    Write-Host "虚拟环境已创建：.venv"
    Write-Host "激活命令：.\\.venv\\Scripts\\Activate.ps1"
    Write-Host "安装依赖：pip install -r real_time_voice_processing\\requirements.txt"
} catch {
    Write-Error "创建虚拟环境失败：$_"
}