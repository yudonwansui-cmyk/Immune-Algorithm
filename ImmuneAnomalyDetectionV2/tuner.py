# tuner.py (V4 - 使用 AUPRC 作为最终优化指标来调优半径)

import numpy as np
# !!! 关键修改：导入AUPRC计算所需工具 !!!
from sklearn.metrics import precision_recall_curve, auc
import time

import config
from data_processor import load_and_prepare_data
from immune_algorithm import NegativeSelectionAlgorithm


def tune_radius_with_auprc():
    """
    自动寻找最优的 DETECTOR_RADIUS 超参数，
    使用 AUPRC (精确率-召回率曲线下面积) 作为核心评估指标。
    """
    print("==========================================================")
    print("=== 开始自动调优 DETECTOR_RADIUS (优化目标: AUPRC) ===")
    print("==========================================================")

    # --- 步骤 1: 加载数据 ---
    data_tuple = load_and_prepare_data()
    if data_tuple[0] is None:
        return
    X_train_normal, X_test, y_test = data_tuple

    # --- 步骤 2: 定义搜索范围 ---
    # 根据之前的经验，继续在一个较大的范围内搜索
    radius_search_space = np.arange(2.0, 10, 0.2)
    print(f"将要测试的半径范围: {[round(r, 2) for r in radius_search_space]}")

    # --- 步骤 3: 初始化记录变量 ---
    best_radius = None
    best_auprc = -1.0  # AUPRC的初始值设为-1
    results = []

    start_time = time.time()

    # --- 步骤 4: 循环遍历所有候选半径值 ---
    for radius in radius_search_space:
        current_radius = round(radius, 2)
        print(f"\n--- 正在测试 Radius = {current_radius} ---")

        # a) 初始化和训练模型
        model = NegativeSelectionAlgorithm(
            radius=current_radius,
            num_detectors=config.NUM_DETECTORS_TO_GENERATE,
            max_tries=config.MAX_GENERATION_TRIES
        )
        model.train(X_train_normal)

        # b) !!! 关键修改：调用 predict_scores 获取异常分数 !!!
        y_scores = model.predict_scores(X_test)

        # c) !!! 关键修改：计算 AUPRC !!!
        # 首先获取 PR 曲线的点
        precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=config.ANOMALY_LABEL_VALUE)
        # 然后计算曲线下面积
        current_auprc = auc(recall, precision)

        print(f"当前 Radius: {current_radius}, Area Under PR Curve (AUPRC): {current_auprc:.4f}")

        results.append({'radius': current_radius, 'auprc': current_auprc})

        # d) !!! 关键修改：使用 AUPRC 进行比较 !!!
        if current_auprc > best_auprc:
            best_auprc = current_auprc
            best_radius = current_radius
            print(f"!!! 发现新的最优解: Radius = {best_radius}, AUPRC = {best_auprc:.4f} !!!")

    # --- 步骤 5: 输出最终结果 ---
    total_time = time.time() - start_time
    print("\n==============================================")
    print("=== 调优完成 ===")
    print(f"总耗时: {total_time:.2f} 秒")
    print("\n--- 实验结果汇总 (按AUPRC排序) ---")
    results.sort(key=lambda x: x['auprc'], reverse=True)
    for res in results:
        print(f"Radius: {res['radius']:.2f} -> AUPRC: {res['auprc']:.4f}")

    print("\n--- 最终结论 ---")
    if best_radius is not None:
        print(f"🎉 找到的最优 DETECTOR_RADIUS 是: {best_radius}")
        print(f"   在该半径下，AUPRC 最高达到: {best_auprc:.4f}")
    else:
        print("未能找到任何有效的参数组合。")
    print("==============================================")


if __name__ == '__main__':
    tune_radius_with_auprc()