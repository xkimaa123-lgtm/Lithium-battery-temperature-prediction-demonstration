@echo off
echo ========================================
echo 人工神经网络基础及应用课程-锂电池温度预测演示系统 安装脚本
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    echo.
    echo 推荐安装方式:
    echo 1. 访问 https://www.python.org/downloads/
    echo 2. 下载Python 3.9或更高版本
    echo 3. 安装时勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo Python版本:
python --version
echo.

REM 升级pip
echo 升级pip...
python -m pip install --upgrade pip

REM 安装依赖
echo.
echo 安装依赖包...
pip install numpy pandas scikit-learn scipy matplotlib joblib

echo.
echo ========================================
echo 安装完成!
echo ========================================
echo.
echo 使用方法:
echo   运行 run.bat 启动应用
echo   或者执行: python main.py
echo.
pause
