"""
配置管理模块
"""
import os

# 应用配置
APP_TITLE = "人工神经网络基础及应用课程-锂电池温度预测演示系统"
APP_VERSION = "2.0"

# 默认超参数
DEFAULT_PARAMS = {
    'hidden1': 128,
    'hidden2': 64,
    'hidden3': 32,
    'max_iter': 500,
    'learning_rate': 0.001,
    'alpha': 0.0001,
}

# 数据配置
DATA_DIR = r"E:\数据\data"
MODEL_SAVE_DIR = "saved_models"
REPORT_DIR = "reports"

# 特征列
FEATURE_COLS = [
    'voltage_v', 'current_a', 'time_s', 'ambient_temperature_c',
    'cycle_index', 'operation_type', 'temperature_lag_1',
    'temperature_lag_2', 'temperature_lag_3'
]

# 模型列表
MODELS = ["标准MLP", "Residual-MLP", "Ridge回归", "RandomForest"]

# 颜色配置
COLORS = {
    'primary': '#2F5597',
    'secondary': '#4472C4',
    'success': '#70AD47',
    'warning': '#ED7D31',
    'info': '#FFC000',
    'background': '#f0f2f6',
}

# 字体配置 - 使用系统兼容字体
FONTS = {
    'title': ('Microsoft YaHei', 20, 'bold'),  # 微软雅黑
    'heading': ('Microsoft YaHei', 12, 'bold'),
    'normal': ('Microsoft YaHei', 10),
    'small': ('Microsoft YaHei', 9),
}

# 确保目录存在
for dir_path in [MODEL_SAVE_DIR, REPORT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
