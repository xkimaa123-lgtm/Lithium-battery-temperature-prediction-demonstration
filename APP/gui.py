"""
主窗口GUI模块
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os

from config import (APP_TITLE, APP_VERSION, DEFAULT_PARAMS, MODELS,
                    COLORS, FONTS, DATA_DIR, FEATURE_COLS)
from data_loader import DataLoader
from model_trainer import ModelTrainer
from validators import DataValidator
from visualizer import Visualizer
from dialogs import (PredictionDialog, CrossValidationDialog,
                     HyperparameterDialog, DataStatsDialog)


class MainWindow:
    """主窗口"""

    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_TITLE} v{APP_VERSION}")
        self.root.geometry("1300x850")
        self.root.configure(bg=COLORS['background'])

        # 初始化组件
        self.data_loader = DataLoader()
        self.model_trainer = ModelTrainer()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        """创建主界面组件"""
        # 标题栏
        self.create_title_bar()

        # 主体框架
        main_frame = tk.Frame(self.root, bg=COLORS['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = tk.Frame(main_frame, bg='white', width=320,
                                relief=tk.RAISED, bd=1)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 右侧图表区域
        self.chart_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=1)
        self.chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建控制面板
        self.create_control_panel(control_frame)

        # 创建图表区域默认内容
        self.create_default_chart()

    def create_title_bar(self):
        """创建标题栏"""
        title_frame = tk.Frame(self.root, bg=COLORS['primary'], height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        tk.Label(title_frame,
                text=f"{APP_TITLE}",
                font=FONTS['title'], fg='white', bg=COLORS['primary']).pack(pady=15)

    def create_control_panel(self, parent):
        """创建控制面板"""
        # 数据操作区域
        self.create_data_section(parent)

        # 模型选择区域
        self.create_model_section(parent)

        # 超参数设置区域
        self.create_params_section(parent)

        # 操作按钮区域
        self.create_action_section(parent)

        # 状态栏
        self.create_status_bar(parent)

    def create_data_section(self, parent):
        """数据操作区域"""
        tk.Label(parent, text="数据操作", font=FONTS['heading'],
                bg='white', fg=COLORS['primary']).pack(pady=(10, 5), padx=10, anchor='w')

        # 第一行：选择数据
        btn_frame1 = tk.Frame(parent, bg='white')
        btn_frame1.pack(fill=tk.X, padx=10, pady=3)

        tk.Button(btn_frame1, text="选择文件", command=self.load_single_file,
                 bg=COLORS['secondary'], fg='white', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

        tk.Button(btn_frame1, text="选择目录", command=self.load_directory,
                 bg=COLORS['success'], fg='white', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

        # 第二行：数据操作
        btn_frame2 = tk.Frame(parent, bg='white')
        btn_frame2.pack(fill=tk.X, padx=10, pady=3)

        tk.Button(btn_frame2, text="数据统计", command=self.show_data_stats,
                 bg=COLORS['info'], fg='black', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

        tk.Button(btn_frame2, text="数据分布", command=self.show_data_distribution,
                 bg=COLORS['warning'], fg='white', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

        self.data_label = tk.Label(parent, text="未加载数据", bg='white',
                                   fg='gray', font=FONTS['small'])
        self.data_label.pack(pady=5, padx=10)

        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)

    def create_model_section(self, parent):
        """模型选择区域"""
        tk.Label(parent, text="模型选择", font=FONTS['heading'],
                bg='white', fg=COLORS['primary']).pack(pady=(0, 5), padx=10, anchor='w')

        self.model_var = tk.StringVar(value="标准MLP")

        for model in MODELS:
            tk.Radiobutton(parent, text=model, variable=self.model_var,
                          value=model, bg='white', font=FONTS['normal'],
                          anchor='w').pack(padx=20, anchor='w')

        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)

    def create_params_section(self, parent):
        """超参数设置区域"""
        tk.Label(parent, text="超参数设置", font=FONTS['heading'],
                bg='white', fg=COLORS['primary']).pack(pady=(0, 5), padx=10, anchor='w')

        params = [
            ("隐藏层1神经元:", 'hidden1', DEFAULT_PARAMS['hidden1']),
            ("隐藏层2神经元:", 'hidden2', DEFAULT_PARAMS['hidden2']),
            ("隐藏层3神经元:", 'hidden3', DEFAULT_PARAMS['hidden3']),
            ("最大迭代次数:", 'max_iter', DEFAULT_PARAMS['max_iter']),
        ]

        self.param_vars = {}
        for label, key, default in params:
            frame = tk.Frame(parent, bg='white')
            frame.pack(fill=tk.X, padx=10, pady=2)

            tk.Label(frame, text=label, bg='white',
                    font=FONTS['small']).pack(side=tk.LEFT)

            var = tk.StringVar(value=str(default))
            tk.Entry(frame, textvariable=var, width=8,
                    font=FONTS['small']).pack(side=tk.RIGHT)
            self.param_vars[key] = var

        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)

    def create_action_section(self, parent):
        """操作按钮区域"""
        # 第一行按钮
        row1 = tk.Frame(parent, bg='white')
        row1.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(row1, text="开始训练", command=self.train_model,
                 bg=COLORS['success'], fg='white', font=FONTS['heading'],
                 width=15).pack(side=tk.LEFT, padx=2)

        tk.Button(row1, text="多模型对比", command=self.compare_models,
                 bg=COLORS['warning'], fg='white', font=FONTS['normal'],
                 width=12).pack(side=tk.LEFT, padx=2)

        # 第二行按钮
        row2 = tk.Frame(parent, bg='white')
        row2.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(row2, text="交互预测", command=self.interactive_predict,
                 bg=COLORS['info'], fg='black', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

        tk.Button(row2, text="保存模型", command=self.save_model,
                 bg='gray', fg='white', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

        tk.Button(row2, text="加载模型", command=self.load_model,
                 bg='gray', fg='white', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

        # 第三行按钮
        row3 = tk.Frame(parent, bg='white')
        row3.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(row3, text="交叉验证", command=self.cross_validate,
                 bg=COLORS['secondary'], fg='white', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

        tk.Button(row3, text="超参数搜索", command=self.hyperparameter_search,
                 bg=COLORS['secondary'], fg='white', font=FONTS['normal'],
                 width=12).pack(side=tk.LEFT, padx=2)

        tk.Button(row3, text="特征重要性", command=self.feature_importance,
                 bg=COLORS['secondary'], fg='white', font=FONTS['normal'],
                 width=10).pack(side=tk.LEFT, padx=2)

    def create_status_bar(self, parent):
        """创建状态栏"""
        self.status_label = tk.Label(parent, text="就绪", bg='white',
                                    fg='green', font=FONTS['small'])
        self.status_label.pack(pady=10)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var,
                                           maximum=100, length=280)
        self.progress_bar.pack(pady=5, padx=10)

    def create_default_chart(self):
        """创建默认图表区域"""
        tk.Label(self.chart_frame,
                text=f"欢迎使用\n{APP_TITLE}",
                font=FONTS['title'], bg='white', fg=COLORS['primary']).pack(pady=50)

        tk.Label(self.chart_frame,
                text="请先加载数据，然后选择模型进行训练",
                font=FONTS['heading'], bg='white', fg='gray').pack(pady=10)

        tk.Label(self.chart_frame,
                text="功能特点:\n"
                     "- 支持多种机器学习模型\n"
                     "- 超参数自动调优\n"
                     "- 时间序列交叉验证\n"
                     "- 特征重要性分析\n"
                     "- 交互式预测\n"
                     "- 模型保存/加载",
                font=FONTS['normal'], bg='white', fg='gray',
                justify=tk.LEFT).pack(pady=20)

    def update_progress(self, value, message=""):
        """更新进度条"""
        self.progress_var.set(value)
        if message:
            self.status_label.config(text=message, fg='orange')
        self.root.update_idletasks()

    def load_single_file(self):
        """选择单个.mat文件加载"""
        filepath = filedialog.askopenfilename(
            title="选择.mat数据文件",
            filetypes=[("MATLAB数据文件", "*.mat"), ("所有文件", "*.*")],
            initialdir=DATA_DIR if os.path.exists(DATA_DIR) else "."
        )
        if filepath:
            self._load_data_from_path(filepath)

    def load_directory(self):
        """选择目录加载所有.mat文件"""
        dirpath = filedialog.askdirectory(
            title="选择数据目录",
            initialdir=DATA_DIR if os.path.exists(DATA_DIR) else "."
        )
        if dirpath:
            self._load_data_from_path(dirpath)

    def load_data(self):
        """加载默认数据目录"""
        self._load_data_from_path(DATA_DIR)

    def _load_data_from_path(self, data_path):
        """从指定路径加载数据"""
        def load_thread():
            try:
                self.df = self.data_loader.load_data(data_path, self.update_progress)
                stats = self.data_loader.get_data_stats()

                # 显示加载的文件信息
                files_info = ""
                if hasattr(self.data_loader, 'loaded_files'):
                    n_files = len(self.data_loader.loaded_files)
                    files_info = f"{n_files}个文件\n"

                self.data_label.config(
                    text=f"已加载: {files_info}{stats['样本数']:,}条样本\n{stats['电池数']}只电池",
                    fg='black'
                )
                self.status_label.config(text="数据加载完成!", fg='green')
                messagebox.showinfo("成功", f"数据加载完成!\n共{len(self.df):,}条样本")
            except Exception as e:
                messagebox.showerror("错误", f"加载数据失败: {str(e)}")
                self.status_label.config(text="加载失败", fg='red')

        threading.Thread(target=load_thread, daemon=True).start()

    def train_model(self):
        """训练模型"""
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return

        def train_thread():
            try:
                self.status_label.config(text="正在准备数据...", fg='orange')
                self.root.update()

                # 准备数据
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    self.model_trainer.prepare_data(self.df, FEATURE_COLS)

                # 获取超参数
                params = {
                    'hidden1': int(self.param_vars['hidden1'].get()),
                    'hidden2': int(self.param_vars['hidden2'].get()),
                    'hidden3': int(self.param_vars['hidden3'].get()),
                    'max_iter': int(self.param_vars['max_iter'].get()),
                }

                # 训练模型
                model_type = self.model_var.get()
                y_test, y_pred, history = self.model_trainer.train_model(
                    self.X_train, self.X_test, self.y_train, self.y_test,
                    model_type, params, self.update_progress
                )

                # 显示结果
                self.show_train_results(y_test, y_pred, history)

                self.status_label.config(text="训练完成!", fg='green')
                messagebox.showinfo("完成",
                    f"训练完成!\nRMSE: {history['rmse']:.4f}°C\n"
                    f"MAE: {history['mae']:.4f}°C\nR²: {history['r2']:.6f}")

            except Exception as e:
                messagebox.showerror("错误", f"训练失败: {str(e)}")
                self.status_label.config(text="训练失败", fg='red')

        threading.Thread(target=train_thread, daemon=True).start()

    def show_train_results(self, y_test, y_pred, history):
        """显示训练结果"""
        # 清除图表区域
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # 结果标题
        tk.Label(self.chart_frame,
                text=f"{history['model_type']} 训练结果",
                font=FONTS['heading'], bg='white', fg=COLORS['primary']).pack(pady=10)

        # 指标显示
        metrics_frame = tk.Frame(self.chart_frame, bg='white')
        metrics_frame.pack(pady=5)

        metrics = [
            ("RMSE", f"{history['rmse']:.4f}°C"),
            ("MAE", f"{history['mae']:.4f}°C"),
            ("R²", f"{history['r2']:.6f}"),
            ("训练时间", f"{history['train_time']:.2f}s"),
        ]

        for name, value in metrics:
            frame = tk.Frame(metrics_frame, bg=COLORS['background'], padx=15, pady=10)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=name, font=FONTS['small'],
                    bg=COLORS['background'], fg='gray').pack()
            tk.Label(frame, text=value, font=FONTS['heading'],
                    bg=COLORS['background'], fg=COLORS['primary']).pack()

        # 绘制图表
        Visualizer.plot_train_results(y_test, y_pred, history, self.chart_frame)

    def compare_models(self):
        """多模型对比"""
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return

        def compare_thread():
            try:
                self.status_label.config(text="正在训练多个模型...", fg='orange')
                self.root.update()

                # 准备数据
                X_train, X_test, y_train, y_test = \
                    self.model_trainer.prepare_data(self.df, FEATURE_COLS)

                results = {}

                for idx, model_type in enumerate(MODELS):
                    self.update_progress(idx / len(MODELS) * 100,
                                       f"训练 {model_type}...")

                    # 创建新的训练器
                    trainer = ModelTrainer()
                    _, _, history = trainer.train_model(
                        X_train, X_test, y_train, y_test, model_type
                    )
                    results[model_type] = history

                # 显示对比结果
                self.show_compare_results(results)

                self.status_label.config(text="对比完成!", fg='green')
                messagebox.showinfo("完成", "多模型对比完成!")

            except Exception as e:
                messagebox.showerror("错误", f"对比失败: {str(e)}")
                self.status_label.config(text="对比失败", fg='red')

        threading.Thread(target=compare_thread, daemon=True).start()

    def show_compare_results(self, results):
        """显示对比结果"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        tk.Label(self.chart_frame, text="多模型对比结果",
                font=FONTS['heading'], bg='white', fg=COLORS['primary']).pack(pady=10)

        # 结果表格
        table_frame = tk.Frame(self.chart_frame, bg='white')
        table_frame.pack(pady=5)

        headers = ["模型", "RMSE (°C)", "MAE (°C)", "R²", "训练时间 (s)"]
        for col, header in enumerate(headers):
            tk.Label(table_frame, text=header, font=FONTS['heading'],
                    bg='white', width=12).grid(row=0, column=col, padx=5)

        for row, (name, history) in enumerate(results.items(), 1):
            values = [
                name,
                f"{history['rmse']:.4f}",
                f"{history['mae']:.4f}",
                f"{history['r2']:.6f}",
                f"{history['train_time']:.2f}",
            ]
            for col, value in enumerate(values):
                tk.Label(table_frame, text=value, font=FONTS['normal'],
                        bg='white', width=12).grid(row=row, column=col, padx=5)

        # 绘制对比图
        Visualizer.plot_compare_results(results, self.chart_frame)

    def interactive_predict(self):
        """交互预测"""
        if self.model_trainer.model is None:
            messagebox.showwarning("警告", "请先训练模型!\n\n"
                                    "训练模型步骤:\n"
                                    "1. 加载数据\n"
                                    "2. 选择模型类型\n"
                                    "3. 点击'开始训练'\n"
                                    "4. 训练完成后即可使用预测功能")
            return

        # 显示模型信息
        model_info = f"当前模型: {self.model_trainer.model_type}\n"
        if hasattr(self.model_trainer, 'training_history') and self.model_trainer.training_history:
            history = self.model_trainer.training_history
            model_info += f"训练RMSE: {history.get('rmse', 'N/A'):.4f}°C\n"
            model_info += f"训练R²: {history.get('r2', 'N/A'):.4f}"

        messagebox.showinfo("预测准备", f"模型已就绪!\n\n{model_info}\n\n"
                           "即将打开预测窗口，请输入电池参数进行温度预测。")

        PredictionDialog(self.root, self.model_trainer, FEATURE_COLS)

    def save_model(self):
        """保存模型"""
        if self.model_trainer.model is None:
            messagebox.showwarning("警告", "请先训练模型!")
            return

        try:
            filepath = self.model_trainer.save_model()
            messagebox.showinfo("成功", f"模型已保存到:\n{filepath}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load_model(self):
        """加载模型"""
        filepath = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            initialdir="saved_models"
        )

        if filepath:
            try:
                self.model_trainer.load_model(filepath)
                self.status_label.config(
                    text=f"已加载: {self.model_trainer.model_type}",
                    fg='green'
                )
                messagebox.showinfo("成功", "模型加载成功!")
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {str(e)}")

    def cross_validate(self):
        """交叉验证"""
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return

        def cv_thread():
            try:
                self.status_label.config(text="正在进行交叉验证...", fg='orange')
                self.root.update()

                # 准备数据
                X_train, X_test, y_train, y_test = \
                    self.model_trainer.prepare_data(self.df, FEATURE_COLS)

                model_type = self.model_var.get()
                results = self.model_trainer.cross_validate(
                    X_train, y_train, model_type, n_splits=5,
                    progress_callback=self.update_progress
                )

                # 显示结果
                dialog = CrossValidationDialog(self.root, self.model_trainer,
                                              X_train, y_train)
                dialog.display_results(results)

                self.status_label.config(text="交叉验证完成!", fg='green')

            except Exception as e:
                messagebox.showerror("错误", f"交叉验证失败: {str(e)}")
                self.status_label.config(text="交叉验证失败", fg='red')

        threading.Thread(target=cv_thread, daemon=True).start()

    def hyperparameter_search(self):
        """超参数搜索"""
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return

        def search_thread():
            try:
                self.status_label.config(text="正在搜索超参数...", fg='orange')
                self.root.update()

                # 准备数据
                X_train, X_test, y_train, y_test = \
                    self.model_trainer.prepare_data(self.df, FEATURE_COLS)

                model_type = self.model_var.get()
                results = self.model_trainer.hyperparameter_search(
                    X_train, y_train, model_type,
                    progress_callback=self.update_progress
                )

                # 显示结果
                dialog = HyperparameterDialog(self.root, self.model_trainer,
                                             X_train, y_train)
                dialog.display_results(results)

                # 更新UI中的参数
                best_params = results['best_params']
                if 'hidden_layer_sizes' in best_params:
                    layers = best_params['hidden_layer_sizes']
                    if len(layers) >= 1:
                        self.param_vars['hidden1'].set(str(layers[0]))
                    if len(layers) >= 2:
                        self.param_vars['hidden2'].set(str(layers[1]))
                    if len(layers) >= 3:
                        self.param_vars['hidden3'].set(str(layers[2]))

                self.status_label.config(text="超参数搜索完成!", fg='green')
                messagebox.showinfo("完成",
                    f"最优RMSE: {results['best_score']:.4f}°C\n"
                    f"已更新参数到界面")

            except Exception as e:
                messagebox.showerror("错误", f"搜索失败: {str(e)}")
                self.status_label.config(text="搜索失败", fg='red')

        threading.Thread(target=search_thread, daemon=True).start()

    def feature_importance(self):
        """特征重要性分析"""
        if self.model_trainer.model is None:
            messagebox.showwarning("警告", "请先训练模型!")
            return

        try:
            if hasattr(self.model_trainer.model, 'feature_importances_'):
                importance = self.model_trainer.model.feature_importances_
            elif hasattr(self.model_trainer.model, 'coef_'):
                importance = np.abs(self.model_trainer.model.coef_)
            else:
                messagebox.showinfo("提示", "当前模型不支持特征重要性分析")
                return

            Visualizer.plot_feature_importance(importance, FEATURE_COLS,
                                              self.chart_frame)
            self.status_label.config(text="特征重要性分析完成!", fg='green')

        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")

    def show_data_stats(self):
        """显示数据统计"""
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return

        DataStatsDialog(self.root, self.df)

    def show_data_distribution(self):
        """显示数据分布"""
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return

        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        tk.Label(self.chart_frame, text="数据分布",
                font=FONTS['heading'], bg='white', fg=COLORS['primary']).pack(pady=10)

        columns = ['voltage_v', 'current_a', 'temperature_c', 'time_s']
        Visualizer.plot_data_distribution(self.df, columns, self.chart_frame)

        self.status_label.config(text="数据分布显示完成!", fg='green')


def main():
    """主函数"""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
