"""
模型训练模块
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import time

from config import MODEL_SAVE_DIR, FEATURE_COLS


class ModelTrainer:
    """模型训练器"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_type = None
        self.training_history = {}

    def prepare_data(self, df, feature_cols=None, test_size=0.2):
        """准备训练数据"""
        if feature_cols is None:
            feature_cols = FEATURE_COLS

        X = df[feature_cols].values
        y = df['temperature_c'].values

        # 移除NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """特征缩放"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, X_test, y_train, y_test,
                    model_type='标准MLP', params=None, progress_callback=None):
        """训练模型"""
        self.model_type = model_type
        start_time = time.time()

        # 特征缩放（RandomForest不需要）
        if model_type != 'RandomForest':
            X_train_s, X_test_s = self.scale_features(X_train, X_test)
        else:
            X_train_s, X_test_s = X_train, X_test

        if progress_callback:
            progress_callback(10, "正在初始化模型...")

        # 创建模型
        if model_type == '标准MLP':
            self.model = self._create_mlp(params)
            if progress_callback:
                progress_callback(20, "正在训练MLP...")
            self.model.fit(X_train_s, y_train)
            if progress_callback:
                progress_callback(80, "MLP训练完成!")

        elif model_type == 'Residual-MLP':
            self.model = self._create_mlp(params)
            if progress_callback:
                progress_callback(20, "正在计算残差...")
            # Residual-MLP: 训练残差
            lag1_train = X_train[:, 6]  # temperature_lag_1
            residual_train = y_train - lag1_train
            if progress_callback:
                progress_callback(30, "正在训练Residual-MLP...")
            self.model.fit(X_train_s, residual_train)
            if progress_callback:
                progress_callback(80, "Residual-MLP训练完成!")

        elif model_type == 'Ridge回归':
            self.model = Ridge(alpha=params.get('alpha', 1.0) if params else 1.0)
            if progress_callback:
                progress_callback(20, "正在训练Ridge回归...")
            self.model.fit(X_train_s, y_train)
            if progress_callback:
                progress_callback(80, "Ridge回归训练完成!")

        elif model_type == 'RandomForest':
            self.model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100) if params else 100,
                random_state=42,
                n_jobs=-1
            )
            if progress_callback:
                progress_callback(20, "正在训练RandomForest...")
            self.model.fit(X_train, y_train)
            if progress_callback:
                progress_callback(80, "RandomForest训练完成!")

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 预测
        if progress_callback:
            progress_callback(85, "正在预测...")

        if model_type == 'Residual-MLP':
            lag1_test = X_test[:, 6]
            y_pred = self.model.predict(X_test_s) + lag1_test
        else:
            y_pred = self.model.predict(X_test_s)

        # 计算指标
        if progress_callback:
            progress_callback(90, "正在计算指标...")

        train_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 存储训练历史
        self.training_history = {
            'model_type': model_type,
            'train_time': train_time,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'train_size': len(X_train),
            'test_size': len(X_test),
        }

        if progress_callback:
            progress_callback(100, f"训练完成! RMSE: {rmse:.4f}°C")

        return y_test, y_pred, self.training_history

    def _create_mlp(self, params=None):
        """创建MLP模型"""
        if params is None:
            params = {}

        return MLPRegressor(
            hidden_layer_sizes=(
                params.get('hidden1', 128),
                params.get('hidden2', 64),
                params.get('hidden3', 32)
            ),
            activation='relu',
            solver='adam',
            learning_rate_init=params.get('learning_rate', 0.001),
            alpha=params.get('alpha', 0.0001),
            max_iter=params.get('max_iter', 500),
            early_stopping=True,
            random_state=42,
            verbose=False
        )

    def cross_validate(self, X, y, model_type='标准MLP', n_splits=5,
                       progress_callback=None):
        """时间序列交叉验证"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {'rmse': [], 'mae': [], 'r2': []}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if progress_callback:
                progress = (fold + 1) / n_splits * 100
                progress_callback(progress, f"交叉验证 Fold {fold+1}/{n_splits}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 特征缩放
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # 训练模型
            if model_type == 'RandomForest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model = self._create_mlp()
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)

            scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            scores['mae'].append(mean_absolute_error(y_test, y_pred))
            scores['r2'].append(r2_score(y_test, y_pred))

        # 计算平均值和标准差
        results = {
            'rmse_mean': np.mean(scores['rmse']),
            'rmse_std': np.std(scores['rmse']),
            'mae_mean': np.mean(scores['mae']),
            'mae_std': np.std(scores['mae']),
            'r2_mean': np.mean(scores['r2']),
            'r2_std': np.std(scores['r2']),
            'folds': n_splits
        }

        return results

    def hyperparameter_search(self, X_train, y_train, model_type='标准MLP',
                             progress_callback=None):
        """超参数搜索"""
        if progress_callback:
            progress_callback(10, "正在设置参数网格...")

        if model_type in ['标准MLP', 'Residual-MLP']:
            param_grid = {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                'learning_rate_init': [0.001, 0.01],
                'alpha': [0.0001, 0.001],
            }
            model = MLPRegressor(max_iter=300, early_stopping=True, random_state=42)

        elif model_type == 'Ridge回归':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0],
            }
            model = Ridge()

        elif model_type == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 特征缩放
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        if progress_callback:
            progress_callback(20, "正在搜索最优参数...")

        # 网格搜索
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_scaled, y_train)

        if progress_callback:
            progress_callback(90, "搜索完成!")

        results = {
            'best_params': grid_search.best_params_,
            'best_score': np.sqrt(-grid_search.best_score_),
            'cv_results': grid_search.cv_results_
        }

        return results

    def save_model(self, filename=None):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")

        if filename is None:
            filename = f"model_{self.model_type}_{int(time.time())}.pkl"

        filepath = os.path.join(MODEL_SAVE_DIR, filename)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'training_history': self.training_history,
        }

        joblib.dump(model_data, filepath)
        return filepath

    def load_model(self, filepath):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.training_history = model_data.get('training_history', {})

        return self

    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("请先训练或加载模型")

        if self.model_type != 'RandomForest' and self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)
