@echo off
chcp 65001 >nul
echo ========================================
echo   人工神经网络基础及应用课程-锂电池温度预测演示系统 - 快速指南
echo ========================================
echo.

echo 1. 首次使用，请运行 install.bat 安装依赖
echo 2. 然后运行 run.bat 启动应用
echo.
echo 或者手动执行：
echo   pip install -r requirements.txt
echo   python main.py
echo.
echo ========================================
echo   功能介绍
echo ========================================
echo.
echo   [数据操作]
echo   - 加载数据: 加载.mat格式电池数据
echo   - 数据统计: 查看数据集统计信息
echo   - 数据分布: 可视化数据分布情况
echo.
echo   [模型训练]
echo   - 标准MLP: 多层感知器神经网络
echo   - Residual-MLP: 残差MLP改进模型
echo   - Ridge回归: 岭回归模型
echo   - RandomForest: 随机森林模型
echo.
echo   [高级功能]
echo   - 多模型对比: 同时训练4个模型对比性能
echo   - 交叉验证: 时间序列交叉验证
echo   - 超参数搜索: 自动优化模型参数
echo   - 特征重要性: 分析特征贡献度
echo   - 交互预测: 手动输入参数预测
echo   - 模型保存/加载: 保存和恢复训练好的模型
echo.
echo ========================================
echo   性能指标说明
echo ========================================
echo.
echo   RMSE (均方根误差): 预测误差的平均幅度，越小越好
echo   MAE (平均绝对误差): 预测误差的平均绝对值，越小越好
echo   R² (决定系数): 模型解释能力，越接近1越好
echo.
echo ========================================
echo   提示
echo ========================================
echo.
echo   - 第一次加载数据可能需要几分钟
echo   - 训练大模型时请耐心等待
echo   - 可以随时保存训练好的模型
echo   - 建议先用小参数测试，再用大参数训练
echo.
pause
