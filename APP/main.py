"""
人工神经网络基础及应用课程-锂电池温度预测演示系统
版本: 2.0
功能:
- 多种机器学习模型支持
- 超参数自动调优
- 时间序列交叉验证
- 特征重要性分析
- 交互式预测
- 模型保存/加载
- 数据验证和统计
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import main

if __name__ == "__main__":
    main()
