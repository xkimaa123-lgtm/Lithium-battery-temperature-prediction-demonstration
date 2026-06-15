"""
可视化模块
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化工具"""

    # 颜色配置
    COLORS = {
        'primary': '#4472C4',
        'secondary': '#70AD47',
        'warning': '#ED7D31',
        'info': '#FFC000',
        'danger': '#FF4444',
    }

    @staticmethod
    def plot_train_results(y_test, y_pred, metrics, parent_frame):
        """绘制训练结果"""
        # 清除旧图表
        for widget in parent_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 真实值 vs 预测值散点图
        axes[0].scatter(y_test[:500], y_pred[:500], alpha=0.5, s=15,
                       c=Visualizer.COLORS['primary'])
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0].set_xlabel('真实温度 (°C)', fontsize=11)
        axes[0].set_ylabel('预测温度 (°C)', fontsize=11)
        axes[0].set_title('真实值 vs 预测值', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 误差分布直方图
        errors = y_test - y_pred
        axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7,
                    color=Visualizer.COLORS['secondary'])
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('预测误差 (°C)', fontsize=11)
        axes[1].set_ylabel('频数', fontsize=11)
        axes[1].set_title('误差分布', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"均值: {np.mean(errors):.4f}°C\n标准差: {np.std(errors):.4f}°C"
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        plt.close(fig)
        return canvas

    @staticmethod
    def plot_compare_results(results, parent_frame):
        """绘制模型对比图"""
        for widget in parent_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(results.keys())
        rmse_vals = [r['rmse'] for r in results.values()]
        colors = [
            Visualizer.COLORS['primary'],
            Visualizer.COLORS['secondary'],
            Visualizer.COLORS['warning'],
            Visualizer.COLORS['info'],
        ]

        bars = ax.bar(models, rmse_vals, color=colors[:len(models)], width=0.6)

        # 添加数值标签
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('RMSE (°C)', fontsize=12)
        ax.set_title('不同模型RMSE对比', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(rmse_vals) * 1.2)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        plt.close(fig)
        return canvas

    @staticmethod
    def plot_learning_curve(train_losses, val_losses, parent_frame):
        """绘制学习曲线"""
        for widget in parent_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(10, 5))

        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('损失 (MSE)', fontsize=11)
        ax.set_title('学习曲线', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        plt.close(fig)
        return canvas

    @staticmethod
    def plot_feature_importance(importance, feature_names, parent_frame):
        """绘制特征重要性"""
        for widget in parent_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(10, 6))

        # 排序
        indices = np.argsort(importance)
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = importance[indices]

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_features)))

        ax.barh(range(len(sorted_features)), sorted_importance, color=colors)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel('重要性', fontsize=11)
        ax.set_title('特征重要性', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        plt.close(fig)
        return canvas

    @staticmethod
    def plot_data_distribution(df, columns, parent_frame):
        """绘制数据分布图"""
        for widget in parent_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(columns):
            if col in df.columns:
                axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7,
                             color=Visualizer.COLORS['primary'])
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('频数', fontsize=10)
                axes[idx].set_title(f'{col} 分布', fontsize=11, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)

        # 隐藏空的子图
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        plt.close(fig)
        return canvas

    @staticmethod
    def plot_correlation_matrix(df, columns, parent_frame):
        """绘制相关性矩阵"""
        for widget in parent_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()

        # 计算相关性矩阵
        corr_matrix = df[columns].corr()

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # 添加数值标签
        for i in range(len(columns)):
            for j in range(len(columns)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', fontsize=9)

        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(columns, fontsize=9)
        ax.set_title('特征相关性矩阵', fontsize=13, fontweight='bold')

        plt.colorbar(im, ax=ax, label='相关系数')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        plt.close(fig)
        return canvas
