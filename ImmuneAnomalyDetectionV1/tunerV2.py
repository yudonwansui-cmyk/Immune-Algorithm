import numpy as np
from sklearn.metrics import f1_score
import time

import config
from data_processor import load_and_prepare_data
from immune_algorithm import NegativeSelectionAlgorithm


def tune_num_detectors():
    """
    在固定最优半径的基础上，自动寻找最优的 NUM_DETECTORS_TO_GENERATE 超参数。
    """
    print("======================================================")
    print("=== 开始自动调优 NUM_DETECTORS_TO_GENERATE 参数 ===")
    print("======================================================")

    # --- 步骤 1: 加载数据 ---
    data_tuple = load_and_prepare_data()
    if data_tuple[0] is None:
        print("\n数据加载失败，程序终止。")
        return
    X_train_normal, X_test, y_test = data_tuple

    # --- 步骤 2: 固定最优半径并定义检测器数量的搜索范围 ---
    # !!! 关键步骤: 使用你刚才找到的最优半径 !!!
    fixed_best_radius = 2.25
    print(f"将使用固定的最优半径: {fixed_best_radius}")

    # 定义搜索空间：从500开始，每次增加50，直到1500（可以根据需要调整上限）
    num_detectors_search_space = range(500, 3001, 50)
    print(f"将要测试的检测器数量范围: {list(num_detectors_search_space)}")

    # --- 步骤 3: 初始化记录变量 ---
    best_num_detectors = None
    best_f1_score = -1.0
    results = []

    start_time = time.time()

    # --- 步骤 4: 循环遍历所有候选的检测器数量 ---
    for num_detectors in num_detectors_search_space:
        print(f"\n--- 正在测试 Num_Detectors = {num_detectors} ---")

        # a) 使用当前参数初始化模型
        model = NegativeSelectionAlgorithm(
            radius=fixed_best_radius,  # 使用固定的最优半径
            num_detectors=num_detectors,  # 使用当前循环的数量
            max_tries=config.MAX_GENERATION_TRIES
        )

        # b) 训练
        model.train(X_train_normal)

        # c) 预测
        y_pred = model.predict(X_test)

        # d) 计算 F1-score
        current_f1 = f1_score(y_test, y_pred, pos_label=config.ANOMALY_LABEL_VALUE)
        print(f"当前 Num_Detectors: {num_detectors}, Anomaly F1-Score: {current_f1:.4f}")

        results.append({'num_detectors': num_detectors, 'f1_score': current_f1})

        # e) 检查是否是新的最优解
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_num_detectors = num_detectors
            print(f"!!! 发现新的最优解: Num_Detectors = {best_num_detectors}, F1-Score = {best_f1_score:.4f} !!!")

    # --- 步骤 5: 输出最终结果 ---
    total_time = time.time() - start_time
    print("\n==============================================")
    print("=== 调优完成 ===")
    print(f"总耗时: {total_time:.2f} 秒")
    print("\n--- 实验结果汇总 ---")
    for res in results:
        print(f"Num_Detectors: {res['num_detectors']:<5} -> Anomaly F1-Score: {res['f1_score']:.4f}")

    print("\n--- 最终结论 ---")
    if best_num_detectors is not None:
        print(f"🎉 在 Radius={fixed_best_radius} 的基础上,")
        print(f"   找到的最优 NUM_DETECTORS_TO_GENERATE 是: {best_num_detectors}")
        print(f"   在该组合下，异常类别的 F1-Score 最高达到: {best_f1_score:.4f}")
    else:
        print("未能找到任何有效的参数组合。")
    print("==============================================")


if __name__ == '__main__':
    tune_num_detectors()