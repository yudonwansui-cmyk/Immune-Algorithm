# 1. 导入所有必要的模块和类
import config
from data_processor import load_and_prepare_data
from immune_algorithm import NegativeSelectionAlgorithm
from evaluation import evaluate_model_performance
import time


def run_experiment():
    """
    主执行函数，串联整个实验流程。
    从数据加载到模型评估，按顺序执行所有步骤。
    """
    print("==============================================")
    print("=== 人工免疫负选择算法进行异常检测实验 ===")
    print("==============================================")

    start_time = time.time()  # 记录实验开始时间

    # --- 步骤 1: 加载和预处理数据 ---
    # 调用 data_processor 模块中的函数
    # 这个函数会返回训练所需的正常样本集，以及用于评估的测试集
    data_tuple = load_and_prepare_data()

    # 检查数据是否加载成功
    if data_tuple[0] is None:
        print("\n数据加载失败，程序终止。")
        return

    X_train_normal, X_test, y_test = data_tuple

    # --- 步骤 2: 初始化并训练模型 ---
    # 从 config 文件中读取算法的超参数
    # 创建 NegativeSelectionAlgorithm 类的实例
    model = NegativeSelectionAlgorithm(
        radius=config.DETECTOR_RADIUS,
        num_detectors=config.NUM_DETECTORS_TO_GENERATE,
        max_tries=config.MAX_GENERATION_TRIES
    )

    # 调用模型的 train 方法，传入正常样本集进行训练
    model.train(X_train_normal)

    # --- 步骤 3: 在测试集上进行预测 ---
    # 调用模型的 predict 方法，对测试集进行预测
    y_pred = model.predict(X_test)

    # --- 步骤 4: 评估模型性能 ---
    # 调用 evaluation 模块中的函数，传入真实标签和预测标签
    evaluate_model_performance(y_test, y_pred)

    # --- 实验结束 ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- 实验流程执行完毕 ---")
    print(f"总耗时: {total_time:.2f} 秒")
    print("==============================================")


# Python程序的标准入口点
# 当你直接运行 `main.py` 文件时，下面的代码块会被执行
if __name__ == '__main__':
    run_experiment()