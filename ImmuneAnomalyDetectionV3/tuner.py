# tuner.py (标准版 - 用于调优半径，优化目标为 F1-Score)

import numpy as np
from sklearn.metrics import f1_score
import time

import config
from data_processor import load_and_prepare_data
from immune_algorithm import NegativeSelectionAlgorithm


def find_best_radius():
    """
    自动寻找最优的 DETECTOR_RADIUS 超参数，
    使用 Anomaly F1-Score 作为核心评估指标。
    """
    print("==========================================================")
    print("=== 开始自动调优 DETECTOR_RADIUS (优化目标: F1-Score) ===")
    print("==========================================================")

    # --- 步骤 1: 加载数据 ---
    data_tuple = load_and_prepare_data()
    if data_tuple[0] is None:
        return
    X_train_normal, X_test, y_test = data_tuple

    # --- 步骤 2: 定义搜索范围 ---
    # 葡萄酒数据集是11维，最优半径可能不大。我们从一个较小的范围开始精细搜索。
    radius_search_space = np.arange(0.1, 4.01, 0.05)
    print(f"将要测试的半径范围: {[round(r, 2) for r in radius_search_space]}")

    # --- 步骤 3: 初始化记录变量 ---
    best_radius = None
    best_f1_score = -1.0
    results = []

    # 从config中获取固定的检测器数量，用于本次调优
    num_detectors_for_tuning = config.NUM_DETECTORS_TO_GENERATE

    start_time = time.time()

    # --- 步骤 4: 循环遍历所有候选半径值 ---
    for radius in radius_search_space:
        current_radius = round(radius, 2)
        print(f"\n--- 正在测试 Radius = {current_radius} ---")

        model = NegativeSelectionAlgorithm(
            radius=current_radius,
            num_detectors=num_detectors_for_tuning,
            max_tries=config.MAX_GENERATION_TRIES
        )
        model.train(X_train_normal)

        # 使用硬分类 predict 方法
        y_pred = model.predict(X_test)

        # 计算 Anomaly F1-Score
        current_f1 = f1_score(y_test, y_pred, pos_label=config.ANOMALY_LABEL_VALUE, zero_division=0)

        print(f"当前 Radius: {current_radius}, Anomaly F1-Score: {current_f1:.4f}")

        results.append({'radius': current_radius, 'f1_score': current_f1})

        # 使用 F1-Score 进行比较
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_radius = current_radius
            print(f"!!! 发现新的最优解: Radius = {best_radius}, F1-Score = {best_f1_score:.4f} !!!")

    # --- 步骤 5: 输出最终结果 ---
    total_time = time.time() - start_time
    print("\n==============================================")
    print("=== 半径调优完成 ===")
    print(f"总耗时: {total_time:.2f} 秒")
    print("\n--- 实验结果汇总 (按F1-Score排序) ---")
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    for res in results:
        print(f"Radius: {res['radius']:.2f} -> Anomaly F1-Score: {res['f1_score']:.4f}")

    print("\n--- 最终结论 ---")
    if best_radius is not None and best_f1_score > 0:
        print(f"🎉 找到的最优 DETECTOR_RADIUS 是: {best_radius}")
        print(f"   在该半径下，Anomaly F1-Score 最高达到: {best_f1_score:.4f}")
    else:
        print("在当前搜索范围内未能找到有效的参数组合 (F1-Score > 0)。请尝试调整搜索范围或其它参数。")
    print("==============================================")


# 确保主程序入口调用的是正确的函数
if __name__ == '__main__':
    find_best_radius()