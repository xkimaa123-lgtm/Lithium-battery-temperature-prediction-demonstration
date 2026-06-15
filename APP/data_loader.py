"""
数据加载和预处理模块
支持加载单个.mat文件或批量加载目录中的所有.mat文件
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import os


class DataLoader:
    """数据加载器"""

    def __init__(self):
        self.df = None
        self.loaded_files = []

    def load_data(self, data_dir, progress_callback=None):
        """加载所有.mat文件（支持目录或单个文件）"""
        # 判断是目录还是单个文件
        if os.path.isfile(data_dir):
            return self.load_single_file(data_dir, progress_callback)
        elif os.path.isdir(data_dir):
            return self.load_directory(data_dir, progress_callback)
        else:
            raise FileNotFoundError(f"路径不存在: {data_dir}")

    def load_single_file(self, filepath, progress_callback=None):
        """加载单个.mat文件"""
        if not filepath.endswith('.mat'):
            raise ValueError("请选择.mat格式的文件")

        if progress_callback:
            progress_callback(10, f"加载 {os.path.basename(filepath)}...")

        filename = os.path.basename(filepath)
        df = self.load_mat_file(filepath, filename)

        if len(df) == 0:
            raise ValueError(f"文件 {filename} 中未加载到数据")

        if progress_callback:
            progress_callback(50, "处理数据...")

        self.df = df
        self.loaded_files = [filename]
        self._post_process(progress_callback)

        return self.df

    def load_directory(self, data_dir, progress_callback=None):
        """加载目录中的所有.mat文件"""
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        all_dfs = []
        self.loaded_files = []
        mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

        if not mat_files:
            raise ValueError("目录中未找到.mat文件")

        for idx, fname in enumerate(mat_files):
            fpath = os.path.join(data_dir, fname)
            df = self.load_mat_file(fpath, fname)
            if len(df) > 0:
                all_dfs.append(df)
                self.loaded_files.append(fname)

            if progress_callback:
                progress = (idx + 1) / len(mat_files) * 80
                progress_callback(progress, f"加载 {fname} ({idx+1}/{len(mat_files)})...")

        if not all_dfs:
            raise ValueError("未加载到任何数据")

        self.df = pd.concat(all_dfs, ignore_index=True)
        self._post_process(progress_callback)

        return self.df

    def _post_process(self, progress_callback=None):
        """数据后处理：排序、创建滞后特征"""
        if progress_callback:
            progress_callback(85, "排序数据...")

        self.df = self.df.sort_values(
            ['battery', 'cycle_index', 'operation_type', 'time_s']
        ).reset_index(drop=True)

        # 创建循环唯一标识
        self.df['cycle_uid'] = (
            self.df['battery'] + '_c' + self.df['cycle_index'].astype(str)
        )

        if progress_callback:
            progress_callback(90, "创建滞后特征...")

        # 构造滞后特征
        self.df = self.create_lag_features(self.df)

        if progress_callback:
            progress_callback(100, f"数据加载完成! 共 {len(self.df):,} 条样本")

    def load_mat_file(self, filepath, filename):
        """加载单个.mat文件"""
        try:
            mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        except Exception:
            mat = sio.loadmat(filepath)

        battery_name = os.path.splitext(filename)[0]
        records = []

        if battery_name not in mat:
            return pd.DataFrame()

        battery = mat[battery_name]
        try:
            cycles = battery.cycle
        except AttributeError:
            return pd.DataFrame()

        if not hasattr(cycles, '__len__'):
            cycles = [cycles]

        for cycle_idx, cycle in enumerate(cycles):
            try:
                cycle_type = cycle.type
            except AttributeError:
                continue

            if cycle_type not in ['charge', 'discharge']:
                continue

            try:
                data = cycle.data
                time_arr = np.array(data.Time).flatten()
                voltage_arr = np.array(data.Voltage_measured).flatten()
                current_arr = np.array(data.Current_measured).flatten()
                temp_arr = np.array(data.Temperature_measured).flatten()
            except (AttributeError, TypeError):
                continue

            n_points = len(time_arr)
            if n_points < 10:
                continue

            # 限制数据点数量
            max_points = 200
            if n_points > max_points:
                indices = np.linspace(0, n_points - 1, max_points, dtype=int)
                time_arr = time_arr[indices]
                voltage_arr = voltage_arr[indices]
                current_arr = current_arr[indices]
                temp_arr = temp_arr[indices]

            # 环境温度
            ambient_temp = 24.0
            try:
                ambient_temp = float(data.Ambient_temperature)
            except (AttributeError, TypeError):
                pass

            for i in range(len(time_arr)):
                records.append({
                    'battery': battery_name,
                    'cycle_index': cycle_idx,
                    'operation_type': 1 if cycle_type == 'charge' else 0,
                    'time_s': float(time_arr[i]),
                    'voltage_v': float(voltage_arr[i]),
                    'current_a': float(current_arr[i]),
                    'ambient_temperature_c': ambient_temp,
                    'temperature_c': float(temp_arr[i]),
                })

        return pd.DataFrame(records)

    def create_lag_features(self, df, n_lags=3):
        """创建滞后特征"""
        for lag in range(1, n_lags + 1):
            df[f'temperature_lag_{lag}'] = df.groupby('cycle_uid')['temperature_c'].shift(lag)

        # 删除含有NaN的行
        lag_cols = [f'temperature_lag_{i}' for i in range(1, n_lags + 1)]
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

        return df

    def get_data_stats(self):
        """获取数据统计信息"""
        if self.df is None:
            return None

        stats = {
            '样本数': len(self.df),
            '电池数': self.df['battery'].nunique(),
            '温度范围': f"{self.df['temperature_c'].min():.1f} - {self.df['temperature_c'].max():.1f} °C",
            '电压范围': f"{self.df['voltage_v'].min():.2f} - {self.df['voltage_v'].max():.2f} V",
            '充电样本': (self.df['operation_type'] == 1).sum(),
            '放电样本': (self.df['operation_type'] == 0).sum(),
        }
        return stats
