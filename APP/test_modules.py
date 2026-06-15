"""
测试脚本 - 验证所有模块可以正常导入
"""

def test_imports():
    """测试所有模块导入"""
    print("=" * 50)
    print("测试模块导入")
    print("=" * 50)

    modules = [
        ('config', 'from config import APP_TITLE, MODELS'),
        ('data_loader', 'from data_loader import DataLoader'),
        ('model_trainer', 'from model_trainer import ModelTrainer'),
        ('validators', 'from validators import DataValidator'),
        ('visualizer', 'from visualizer import Visualizer'),
        ('dialogs', 'from dialogs import PredictionDialog'),
        ('gui', 'from gui import MainWindow'),
    ]

    all_ok = True
    for module_name, import_stmt in modules:
        try:
            exec(import_stmt)
            print(f"[OK] {module_name:<15} - 导入成功")
        except Exception as e:
            print(f"[FAIL] {module_name:<15} - 导入失败: {str(e)}")
            all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("[OK] 所有模块导入测试通过!")
    else:
        print("[FAIL] 存在导入错误，请检查依赖")

    return all_ok

def test_config():
    """测试配置"""
    print("\n" + "=" * 50)
    print("测试配置")
    print("=" * 50)

    try:
        from config import APP_TITLE, APP_VERSION, MODELS, FEATURE_COLS, COLORS, FONTS

        print(f"应用标题: {APP_TITLE}")
        print(f"应用版本: {APP_VERSION}")
        print(f"支持的模型: {', '.join(MODELS)}")
        print(f"特征数量: {len(FEATURE_COLS)}")
        print(f"颜色配置: {len(COLORS)} 项")
        print(f"字体配置: {len(FONTS)} 项")

        print(f"\n[OK] 配置测试通过!")
        return True
    except Exception as e:
        print(f"[FAIL] 配置测试失败: {str(e)}")
        return False

def test_data_validator():
    """测试数据验证器"""
    print("\n" + "=" * 50)
    print("测试数据验证器")
    print("=" * 50)

    try:
        import pandas as pd
        from validators import DataValidator

        # 创建测试数据
        test_data = {
            'voltage_v': [3.7, 3.8, 3.6, 4.5],  # 4.5超出范围
            'current_a': [1.5, 2.0, -1.0, 0.5],
            'temperature_c': [30, 35, 25, 90],  # 90超出范围
            'time_s': [100, 200, 300, 400],
        }
        df = pd.DataFrame(test_data)

        # 测试范围验证
        warnings = DataValidator.validate_ranges(df)
        print("范围验证警告:")
        for w in warnings:
            print(f"  - {w}")

        # 测试异常值检测
        outlier_mask = DataValidator.detect_outliers(df)
        print(f"\n异常值数量: {outlier_mask.sum()}")

        # 测试报告生成
        report = DataValidator.generate_report(df)
        print("\n验证报告预览:")
        print(report[:500])

        print("\n[OK] 数据验证器测试通过!")
        return True
    except Exception as e:
        print(f"[FAIL] 数据验证器测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("人工神经网络基础及应用课程-锂电池温度预测演示系统 - 模块测试")
    print("=" * 60)

    results = []
    results.append(("模块导入", test_imports()))
    results.append(("配置", test_config()))
    results.append(("数据验证器", test_data_validator()))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "[OK] 通过" if ok else "[FAIL] 失败"
        print(f"{name:<20} {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n[OK] 所有测试通过! 系统可以正常运行。")
        print("\n启动应用: python main.py")
    else:
        print("\n[FAIL] 存在测试失败，请检查依赖和代码。")

if __name__ == "__main__":
    main()
