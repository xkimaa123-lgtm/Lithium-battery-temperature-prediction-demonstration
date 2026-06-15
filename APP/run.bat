@echo off
echo ========================================
echo 人工神经网络基础及应用课程-锂电池温度预测演示系统 启动脚本
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查依赖是否安装
echo 检查依赖包...
pip show numpy >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
)

echo.
echo 启动应用...
echo.

REM 启动应用
python main.py

if errorlevel 1 (
    echo.
    echo 应用运行出错，请检查错误信息
    pause
)
