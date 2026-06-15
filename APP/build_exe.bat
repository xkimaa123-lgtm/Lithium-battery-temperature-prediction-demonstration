@echo off
echo ========================================
echo 打包锂电池温度预测系统为exe
echo ========================================
echo.

REM 检查PyInstaller是否安装
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo 安装PyInstaller...
    pip install pyinstaller
)

echo 开始打包...
echo.

REM 打包为单个exe文件
pyinstaller --onefile --windowed --name "锂电池温度预测系统" ^
    --add-data "config.py;." ^
    --add-data "data_loader.py;." ^
    --add-data "model_trainer.py;." ^
    --add-data "validators.py;." ^
    --add-data "visualizer.py;." ^
    --add-data "dialogs.py;." ^
    --add-data "gui.py;." ^
    main.py

echo.
if exist "dist\锂电池温度预测系统.exe" (
    echo ========================================
    echo 打包完成!
    echo ========================================
    echo.
    echo 生成的exe文件位置:
    echo   dist\锂电池温度预测系统.exe
    echo.
    echo 注意: 第一次运行可能需要管理员权限
    echo.
) else (
    echo 打包失败，请检查错误信息
)

pause
