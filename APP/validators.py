"""
数据验证模块
"""
import numpy as np
import pandas as pd


class DataValidator:
    """数据验证器"""

    # 合理范围定义
    RANGES = {
        'voltage_v': (2.0, 5.0),      # 电压范围 (V)
        'current_a': (-10.0, 10.0),    # 电流范围 (A)
        'temperature_c': (-20.0, 80.0), # 温度范围 (°C)
        'time_s': (0, 10000),          # 时间范围 (s)
    }

    @classmethod
    def validate_ranges(cls, df):
        """验证数据范围"""
        warnings = []

        for col, (min_val, max_val) in cls.RANGES.items():
            if col in df.columns:
                below = (df[col] < min_val).sum()
                above = (df[col] > max_val).sum()

                if below > 0:
                    warnings.append(f"{col}: {below}个值低于下限 {min_val}")
                if above > 0:
                    warnings.append(f"{col}: {above}个值高于上限 {max_val}")

        return warnings

    @classmethod
    def detect_outliers(cls, df, columns=None, method='iqr', threshold=1.5):
        """检测异常值"""
        if columns is None:
            columns = ['voltage_v', 'current_a', 'temperature_c']

        outlier_mask = pd.Series(False, index=df.index)

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                col_outliers = (df[col] < lower) | (df[col] > upper)

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > threshold

            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_mask = outlier_mask | col_outliers

        return outlier_mask

    @classmethod
    def check_missing_values(cls, df):
        """检查缺失值"""
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) > 0:
            return missing.to_dict()
        return None

    @classmethod
    def validate_features(cls, df, feature_cols):
        """验证特征列是否存在"""
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺失特征列: {missing_cols}")

    @classmethod
    def clean_data(cls, df, remove_outliers=False):
        """清洗数据"""
        df_clean = df.copy()

        # 移除温度异常值
        if remove_outliers:
            outlier_mask = cls.detect_outliers(df_clean, ['temperature_c'])
            df_clean = df_clean[~outlier_mask].reset_index(drop=True)

        # 移除NaN
        df_clean = df_clean.dropna().reset_index(drop=True)

        return df_clean

    @classmethod
    def generate_report(cls, df):
        """生成数据验证报告"""
        report = []
        report.append("=" * 50)
        report.append("数据验证报告")
        report.append("=" * 50)
        report.append(f"\n样本总数: {len(df):,}")
        report.append(f"特征数量: {len(df.columns)}")

        # 检查缺失值
        missing = cls.check_missing_values(df)
        if missing:
            report.append("\n缺失值:")
            for col, count in missing.items():
                report.append(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            report.append("\n无缺失值")

        # 范围验证
        warnings = cls.validate_ranges(df)
        if warnings:
            report.append("\n范围警告:")
            for w in warnings:
                report.append(f"  - {w}")
        else:
            report.append("\n数据范围正常")

        # 统计信息
        report.append("\n数据统计:")
        for col in ['voltage_v', 'current_a', 'temperature_c']:
            if col in df.columns:
                report.append(f"  {col}:")
                report.append(f"    均值: {df[col].mean():.4f}")
                report.append(f"    标准差: {df[col].std():.4f}")
                report.append(f"    最小值: {df[col].min():.4f}")
                report.append(f"    最大值: {df[col].max():.4f}")

        return "\n".join(report)
