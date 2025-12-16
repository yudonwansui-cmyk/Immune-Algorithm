# tuner.py

import numpy as np
from sklearn.metrics import f1_score  # 我们只需要f1_score来进行比较
import time

# 导入我们项目中的其他模块
import config
from data_processor import load_and_prepare_data
from immune_algorithm import NegativeSelectionAlgorithm


def tune_detector_radius():
    """
    自动寻找最优的 DETECTOR_RADIUS 超参数。
    """
    print("==============================================")
    print("=== 开始自动调优 DETECTOR_RADIUS 参数 ===")
    print("==============================================")

    # --- 步骤 1: 加载并准备数据 (只需要一次) ---
    data_tuple = load_and_prepare_data()
    if data_tuple[0] is None:
        print("\n数据加载失败，程序终止。")
        return
    X_train_normal, X_test, y_test = data_tuple

    # --- 步骤 2: 定义搜索范围 ---
    # 从 0.1 到 1.0, 步长为 0.05，这样更精细一些
    radius_search_space = np.arange(0.0, 3.01, 0.05)
    print(f"将要测试的半径范围: {[round(r, 2) for r in radius_search_space]}")

    # --- 步骤 3: 初始化用于记录最优结果的变量 ---
    best_radius = None
    best_f1_score = -1.0  # 初始化为一个无效值
    results = []  # 记录每次的结果

    start_time = time.time()

    # --- 步骤 4: 循环遍历所有候选半径值 ---
    for radius in radius_search_space:
        current_radius = round(radius, 2)  # 处理浮点数精度问题
        print(f"\n--- 正在测试 Radius = {current_radius} ---")

        # a) 使用当前半径初始化模型
        model = NegativeSelectionAlgorithm(
            radius=current_radius,
            num_detectors=config.NUM_DETECTORS_TO_GENERATE,
            max_tries=config.MAX_GENERATION_TRIES
        )

        # b) 训练模型
        model.train(X_train_normal)

        # c) 进行预测
        y_pred = model.predict(X_test)

        # d) 计算异常类别的 F1-score
        # pos_label=config.ANOMALY_LABEL_VALUE 确保我们计算的是异常类(1)的f1分数
        current_f1 = f1_score(y_test, y_pred, pos_label=config.ANOMALY_LABEL_VALUE)
        print(f"当前 Radius: {current_radius}, Anomaly F1-Score: {current_f1:.4f}")

        results.append({'radius': current_radius, 'f1_score': current_f1})

        # e) 检查是否是目前的最优结果
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_radius = current_radius
            print(f"!!! 发现新的最优解: Radius = {best_radius}, F1-Score = {best_f1_score:.4f} !!!")

    # --- 步骤 5: 输出最终的调优结果 ---
    total_time = time.time() - start_time
    print("\n==============================================")
    print("=== 调优完成 ===")
    print(f"总耗时: {total_time:.2f} 秒")
    print("\n--- 实验结果汇总 ---")
    for res in results:
        print(f"Radius: {res['radius']:.2f} -> Anomaly F1-Score: {res['f1_score']:.4f}")

    print("\n--- 最终结论 ---")
    if best_radius is not None:
        print(f"🎉 找到的最优 DETECTOR_RADIUS 是: {best_radius}")
        print(f"   在该半径下，异常类别的 F1-Score 最高达到: {best_f1_score:.4f}")
    else:
        print("未能找到任何有效的参数组合。")
    print("==============================================")


if __name__ == '__main__':
    tune_detector_radius()
