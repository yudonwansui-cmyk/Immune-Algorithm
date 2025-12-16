# main.py (最终版本 - 包含常规评估和AUPRC高级评估)

import config
from data_processor import load_and_prepare_data
from immune_algorithm import NegativeSelectionAlgorithm
# 导入我们创建的两个评估函数
from evaluation import evaluate_model_performance, evaluate_with_auprc
import time


def run_experiment():
    """
    主执行函数，串联整个实验流程。
    该版本整合了两种评估方式：
    1. 基于硬分类结果 (0/1) 的常规评估 (Confusion Matrix, F1, MCC)。
    2. 基于软分数 (异常得分) 的高级评估 (AUPRC)。
    """
    print("==============================================")
    print("=== 人工免疫算法进行异常检测实验 ===")
    print("==============================================")

    start_time = time.time()

    # --- 步骤 1: 加载和预处理数据 ---
    data_tuple = load_and_prepare_data()

    if data_tuple[0] is None:
        print("\n数据加载失败，程序终止。")
        return

    X_train_normal, X_test, y_test = data_tuple

    print(f"\n--- 2. 开始训练模型 (使用config.py中的最优参数) ---")
    print(f"参数配置: Radius = {config.DETECTOR_RADIUS}, Num_Detectors = {config.NUM_DETECTORS_TO_GENERATE}")

    model = NegativeSelectionAlgorithm(
        radius=config.DETECTOR_RADIUS,
        num_detectors=config.NUM_DETECTORS_TO_GENERATE,
        max_tries=config.MAX_GENERATION_TRIES
    )

    model.train(X_train_normal)

    # --- 评估流程 ---

    # --- Part A: 基于“硬分类”的常规评估 ---
    print("\n--- 开始执行常规评估 (基于0/1预测结果) ---")
    # 步骤 3: 进行硬分类预测 (输出 0 或 1)
    y_pred = model.predict(X_test)
    # 步骤 4: 常规性能评估
    evaluate_model_performance(y_test, y_pred)

    # --- Part B: 基于“软分数”的高级AUPRC评估 ---
    print("\n--- 开始执行高级评估 (基于异常分数和AUPRC) ---")
    # 步骤 3a: 计算每个样本的异常分数
    y_scores = model.predict_scores(X_test)
    # 步骤 4a: 进行AUPRC评估并绘制PR曲线
    evaluate_with_auprc(y_test, y_scores)

    # --- 实验结束 ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- 实验流程执行完毕 ---")
    print(f"总耗时: {total_time:.2f} 秒")
    print("==============================================")


# Python程序的标准入口点
if __name__ == '__main__':
    run_experiment()