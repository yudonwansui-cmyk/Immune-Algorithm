# evaluation.py

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 导入config来获取标签的定义，使报告更具可读性
import config


def evaluate_model_performance(y_true, y_pred):
    """
    全面评估模型的性能，并打印详细的报告和可视化的混淆矩阵。

    Args:
        y_true (np.array or list): 真实的标签数组。
        y_pred (np.array or list): 模型预测的标签数组。
    """
    print("\n--- 4. 开始评估模型性能 ---")

    # 定义标签名称，用于报告和图表显示
    target_names = [f'Normal ({config.NORMAL_LABEL_VALUE})', f'Anomaly ({config.ANOMALY_LABEL_VALUE})']

    # 1. 计算并打印整体准确率
    # 准确率 = (正确预测的样本数) / (总样本数)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"整体准确率 (Accuracy): {accuracy:.4f}")

    # 2. 打印详细的分类报告
    #   - Precision (精确率): 在所有被预测为“正”的样本中，有多少是真正的“正”。(TP / (TP + FP))
    #   - Recall (召回率/查全率): 在所有真正的“正”样本中，有多少被成功预测出来了。(TP / (TP + FN))
    #   - F1-score: 精确率和召回率的调和平均数，是综合评价指标。
    print("\n详细分类报告 (Classification Report):")
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        labels=[config.NORMAL_LABEL_VALUE, config.ANOMALY_LABEL_VALUE]  # 确保报告顺序
    )
    print(report)

    # 3. 生成并可视化混淆矩阵
    # 混淆矩阵能清晰地展示每一类别的预测情况:
    #   - True Positives (TP): 真正例 (异常被正确预测为异常)
    #   - True Negatives (TN): 真负例 (正常被正确预测为正常)
    #   - False Positives (FP): 假正例 (正常被错误预测为异常) - "误报"
    #   - False Negatives (FN): 假负例 (异常被错误预测为正常) - "漏报"
    print("生成混淆矩阵可视化图...")
    cm = confusion_matrix(y_true, y_pred, labels=[config.NORMAL_LABEL_VALUE, config.ANOMALY_LABEL_VALUE])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,  # 在格子里显示数字
        fmt='d',  # 数字格式为整数
        cmap='Blues',  # 颜色主题
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.show()  # 显示图表


# (可选) 用于独立测试该模块
if __name__ == '__main__':
    # 模拟一些真实标签和预测标签
    mock_y_true = np.array([config.NORMAL_LABEL_VALUE] * 50 + [config.ANOMALY_LABEL_VALUE] * 20)  # 50个正常, 20个异常
    # 模拟一个还不错的预测结果：
    # - 50个正常样本中，有48个预测正确，2个被误报为异常
    # - 20个异常样本中，有17个预测正确，3个被漏报为正常
    mock_y_pred = np.array(
        [config.NORMAL_LABEL_VALUE] * 48 + [config.ANOMALY_LABEL_VALUE] * 2 +  # 预测的前50个
        [config.ANOMALY_LABEL_VALUE] * 17 + [config.NORMAL_LABEL_VALUE] * 3  # 预测的后20个
    )

    print("--- 开始模块独立测试 ---")
    evaluate_model_performance(mock_y_true, mock_y_pred)
    print("\n预期结果解读:")
    print(" - 混淆矩阵左上角(TN)应为48，右下角(TP)应为17。")
    print(" - 左下角(FN)应为3 (漏报)，右上角(FP)应为2 (误报)。")