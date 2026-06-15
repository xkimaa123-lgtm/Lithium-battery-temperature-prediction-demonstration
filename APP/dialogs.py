"""
GUI对话框模块
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

from config import FONTS, COLORS, FEATURE_COLS


class PredictionDialog:
    """交互预测对话框"""

    # 特征定义：(标签, 默认值, 合理范围, 单位)
    FEATURE_DEFINITIONS = [
        ("电压", "3.7", (2.0, 5.0), "V"),
        ("电流", "1.5", (-10.0, 10.0), "A"),
        ("时间", "500", (0, 10000), "s"),
        ("环境温度", "24", (-20.0, 60.0), "°C"),
        ("循环编号", "50", (0, 1000), ""),
        ("工况类型", "1", (0, 1), "1=充电, 0=放电"),
        ("T(t-1)", "30", (-20.0, 80.0), "°C"),
        ("T(t-2)", "29", (-20.0, 80.0), "°C"),
        ("T(t-3)", "28", (-20.0, 80.0), "°C"),
    ]

    def __init__(self, parent, model_trainer, feature_cols):
        self.parent = parent
        self.model_trainer = model_trainer
        self.feature_cols = feature_cols
        self.prediction_count = 0

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("温度预测")
        self.dialog.geometry("500x750")
        self.dialog.configure(bg='white')
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.inputs = {}
        self.create_widgets()

    def create_widgets(self):
        """创建对话框组件"""
        # 标题
        title_frame = tk.Frame(self.dialog, bg=COLORS['primary'], height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        tk.Label(title_frame, text="温度预测",
                font=FONTS['title'], fg='white', bg=COLORS['primary']).pack(pady=15)

        # 说明
        tk.Label(self.dialog, text="输入电池参数，预测下一时刻温度",
                font=FONTS['normal'], bg='white', fg='gray').pack(pady=5)

        # 输入区域 - 使用带滚动条的框架
        input_container = tk.Frame(self.dialog, bg='white')
        input_container.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # 创建输入字段
        for idx, (label, default, (min_val, max_val), unit) in enumerate(self.FEATURE_DEFINITIONS):
            row_frame = tk.Frame(input_container, bg='white')
            row_frame.pack(fill=tk.X, pady=4)

            # 标签（带范围提示）
            label_text = f"{label} ({unit})" if unit else label
            tk.Label(row_frame, text=label_text, font=FONTS['normal'],
                    bg='white', width=20, anchor='w').pack(side=tk.LEFT)

            # 输入框
            entry = tk.Entry(row_frame, width=12, font=FONTS['normal'])
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, padx=5)

            # 范围提示
            range_text = f"[{min_val}-{max_val}]"
            tk.Label(row_frame, text=range_text, font=FONTS['small'],
                    bg='white', fg='gray').pack(side=tk.LEFT, padx=5)

            self.inputs[label] = (entry, min_val, max_val)

        # 分隔线
        ttk.Separator(self.dialog, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)

        # 按钮区域
        btn_frame = tk.Frame(self.dialog, bg='white')
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="预测温度", command=self.predict,
                 bg=COLORS['success'], fg='white', font=FONTS['heading'],
                 width=15).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="使用示例", command=self.fill_example,
                 bg=COLORS['info'], fg='black', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="清除", command=self.clear,
                 bg=COLORS['warning'], fg='white', font=FONTS['normal'],
                 width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="关闭", command=self.dialog.destroy,
                 bg='gray', fg='white', font=FONTS['normal'],
                 width=8).pack(side=tk.LEFT, padx=5)

        # 结果显示区域
        result_frame = tk.Frame(self.dialog, bg='#f0f8ff', relief=tk.RAISED, bd=1)
        result_frame.pack(pady=10, padx=20, fill=tk.X)

        self.result_label = tk.Label(result_frame, text="等待预测...",
                                    font=('Microsoft YaHei', 16, 'bold'),
                                    bg='#f0f8ff', fg=COLORS['primary'])
        self.result_label.pack(pady=15)

        # 预测统计
        self.stats_label = tk.Label(result_frame, text="",
                                   font=FONTS['small'], bg='#f0f8ff', fg='gray')
        self.stats_label.pack(pady=(0, 10))

        # 预测历史
        history_frame = tk.Frame(self.dialog, bg='white')
        history_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(history_frame, text="预测历史", font=FONTS['heading'],
               bg='white', fg=COLORS['primary']).pack(anchor='w')

        self.history_text = tk.Text(history_frame, height=6, width=50,
                                   font=FONTS['small'], state=tk.DISABLED)
        scrollbar = tk.Scrollbar(history_frame, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def validate_input(self, label, value_str, min_val, max_val):
        """验证输入值"""
        try:
            value = float(value_str)
        except ValueError:
            raise ValueError(f"{label} 必须是数字")

        if value < min_val or value > max_val:
            raise ValueError(f"{label} 必须在 {min_val} 到 {max_val} 之间")

        return value

    def predict(self):
        """执行预测"""
        try:
            # 获取并验证输入值
            values = []
            feature_names = [f[0] for f in self.FEATURE_DEFINITIONS]

            for idx, (label, default, (min_val, max_val), unit) in enumerate(self.FEATURE_DEFINITIONS):
                entry, _, _ = self.inputs[label]
                val_str = entry.get().strip()

                if not val_str:
                    messagebox.showwarning("警告", f"请输入{label}")
                    return

                value = self.validate_input(label, val_str, min_val, max_val)
                values.append(value)

            # 执行预测
            X = np.array([values])
            pred = self.model_trainer.predict(X)[0]

            # 更新计数
            self.prediction_count += 1

            # 显示结果
            result_text = f"预测温度: {pred:.2f}°C"
            self.result_label.config(text=result_text, fg=COLORS['success'])

            # 更新统计
            self.stats_label.config(text=f"第 {self.prediction_count} 次预测")

            # 添加到历史记录
            self.history_text.config(state=tk.NORMAL)

            # 格式化历史记录
            history_entry = f"[{self.prediction_count}] "
            history_entry += f"电压={values[0]:.1f}V, 电流={values[1]:.1f}A, "
            history_entry += f"时间={values[2]:.0f}s, 环温={values[3]:.0f}°C\n"
            history_entry += f"     预测温度: {pred:.2f}°C"
            history_entry += "\n" + "-" * 40 + "\n"

            self.history_text.insert(tk.END, history_entry)
            self.history_text.see(tk.END)
            self.history_text.config(state=tk.DISABLED)

        except ValueError as e:
            messagebox.showerror("输入错误", str(e))
        except Exception as e:
            messagebox.showerror("预测失败", f"预测过程中发生错误:\n{str(e)}")

    def fill_example(self):
        """填充示例数据"""
        examples = [
            ("电压", "3.7"),
            ("电流", "1.5"),
            ("时间", "500"),
            ("环境温度", "24"),
            ("循环编号", "50"),
            ("工况类型", "1"),
            ("T(t-1)", "30"),
            ("T(t-2)", "29"),
            ("T(t-3)", "28"),
        ]

        for label, value in examples:
            entry, _, _ = self.inputs[label]
            entry.delete(0, tk.END)
            entry.insert(0, value)

        messagebox.showinfo("提示", "已填充示例数据，点击'预测温度'查看结果")

    def clear(self):
        """清除输入"""
        for label, (entry, _, _) in self.inputs.items():
            entry.delete(0, tk.END)
        self.result_label.config(text="等待预测...", fg=COLORS['primary'])


class CrossValidationDialog:
    """交叉验证对话框"""

    def __init__(self, parent, model_trainer, X, y):
        self.parent = parent
        self.model_trainer = model_trainer
        self.X = X
        self.y = y

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("交叉验证结果")
        self.dialog.geometry("500x400")
        self.dialog.configure(bg='white')
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()

    def create_widgets(self):
        """创建对话框组件"""
        tk.Label(self.dialog, text="时间序列交叉验证结果",
                font=FONTS['heading'], bg='white', fg=COLORS['primary']).pack(pady=10)

        # 结果显示区域
        self.result_text = tk.Text(self.dialog, height=15, width=50, font=FONTS['normal'])
        self.result_text.pack(pady=10, padx=20)

        # 关闭按钮
        tk.Button(self.dialog, text="关闭", command=self.dialog.destroy,
                 bg='gray', fg='white', font=FONTS['normal']).pack(pady=10)

    def display_results(self, results):
        """显示结果"""
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, "=" * 40 + "\n")
        self.result_text.insert(tk.END, "交叉验证结果\n")
        self.result_text.insert(tk.END, "=" * 40 + "\n\n")

        self.result_text.insert(tk.END, f"折数: {results['folds']}\n\n")

        self.result_text.insert(tk.END, "RMSE:\n")
        self.result_text.insert(tk.END, f"  均值: {results['rmse_mean']:.4f}°C\n")
        self.result_text.insert(tk.END, f"  标准差: {results['rmse_std']:.4f}°C\n\n")

        self.result_text.insert(tk.END, "MAE:\n")
        self.result_text.insert(tk.END, f"  均值: {results['mae_mean']:.4f}°C\n")
        self.result_text.insert(tk.END, f"  标准差: {results['mae_std']:.4f}°C\n\n")

        self.result_text.insert(tk.END, "R²:\n")
        self.result_text.insert(tk.END, f"  均值: {results['r2_mean']:.6f}\n")
        self.result_text.insert(tk.END, f"  标准差: {results['r2_std']:.6f}\n")


class HyperparameterDialog:
    """超参数搜索对话框"""

    def __init__(self, parent, model_trainer, X_train, y_train):
        self.parent = parent
        self.model_trainer = model_trainer
        self.X_train = X_train
        self.y_train = y_train

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("超参数搜索结果")
        self.dialog.geometry("500x400")
        self.dialog.configure(bg='white')
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()

    def create_widgets(self):
        """创建对话框组件"""
        tk.Label(self.dialog, text="超参数搜索结果",
                font=FONTS['heading'], bg='white', fg=COLORS['primary']).pack(pady=10)

        # 结果显示区域
        self.result_text = tk.Text(self.dialog, height=15, width=50, font=FONTS['normal'])
        self.result_text.pack(pady=10, padx=20)

        # 关闭按钮
        tk.Button(self.dialog, text="关闭", command=self.dialog.destroy,
                 bg='gray', fg='white', font=FONTS['normal']).pack(pady=10)

    def display_results(self, results):
        """显示结果"""
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, "=" * 40 + "\n")
        self.result_text.insert(tk.END, "超参数搜索结果\n")
        self.result_text.insert(tk.END, "=" * 40 + "\n\n")

        self.result_text.insert(tk.END, "最优参数:\n")
        for param, value in results['best_params'].items():
            self.result_text.insert(tk.END, f"  {param}: {value}\n")

        self.result_text.insert(tk.END, f"\n最优RMSE: {results['best_score']:.4f}°C\n")


class DataStatsDialog:
    """数据统计对话框"""

    def __init__(self, parent, df):
        self.parent = parent
        self.df = df

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("数据统计信息")
        self.dialog.geometry("400x350")
        self.dialog.configure(bg='white')
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()

    def create_widgets(self):
        """创建对话框组件"""
        tk.Label(self.dialog, text="数据统计信息",
                font=FONTS['heading'], bg='white', fg=COLORS['primary']).pack(pady=10)

        # 统计信息显示
        self.stats_text = tk.Text(self.dialog, height=15, width=45, font=FONTS['normal'])
        self.stats_text.pack(pady=10, padx=20)

        # 计算统计信息
        self.display_stats()

        # 关闭按钮
        tk.Button(self.dialog, text="关闭", command=self.dialog.destroy,
                 bg='gray', fg='white', font=FONTS['normal']).pack(pady=10)

    def display_stats(self):
        """显示统计信息"""
        self.stats_text.delete(1.0, tk.END)

        self.stats_text.insert(tk.END, "=" * 35 + "\n")
        self.stats_text.insert(tk.END, "数据集统计\n")
        self.stats_text.insert(tk.END, "=" * 35 + "\n\n")

        self.stats_text.insert(tk.END, f"样本总数: {len(self.df):,}\n")
        self.stats_text.insert(tk.END, f"电池数量: {self.df['battery'].nunique()}\n")
        self.stats_text.insert(tk.END, f"特征数量: {len(self.df.columns)}\n\n")

        self.stats_text.insert(tk.END, "温度统计:\n")
        self.stats_text.insert(tk.END, f"  范围: {self.df['temperature_c'].min():.1f} - {self.df['temperature_c'].max():.1f}°C\n")
        self.stats_text.insert(tk.END, f"  均值: {self.df['temperature_c'].mean():.2f}°C\n")
        self.stats_text.insert(tk.END, f"  标准差: {self.df['temperature_c'].std():.2f}°C\n\n")

        self.stats_text.insert(tk.END, "电压统计:\n")
        self.stats_text.insert(tk.END, f"  范围: {self.df['voltage_v'].min():.2f} - {self.df['voltage_v'].max():.2f}V\n")
        self.stats_text.insert(tk.END, f"  均值: {self.df['voltage_v'].mean():.2f}V\n\n")

        self.stats_text.insert(tk.END, "电流统计:\n")
        self.stats_text.insert(tk.END, f"  范围: {self.df['current_a'].min():.2f} - {self.df['current_a'].max():.2f}A\n")
        self.stats_text.insert(tk.END, f"  均值: {self.df['current_a'].mean():.2f}A\n\n")

        # 充电/放电样本数
        charge_count = (self.df['operation_type'] == 1).sum()
        discharge_count = (self.df['operation_type'] == 0).sum()
        self.stats_text.insert(tk.END, f"充电样本: {charge_count:,}\n")
        self.stats_text.insert(tk.END, f"放电样本: {discharge_count:,}\n")
