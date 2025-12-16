# evaluation.py (V3 - 增加不平衡数据集的评价指标)

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                                 balanced_accuracy_score, matthews_corrcoef,
                                 precision_recall_curve, auc)
import seaborn as sns
import matplotlib.pyplot as plt
import config


def evaluate_model_performance(y_true, y_pred):
    """
    全面评估模型的性能，特别增加了适用于不平衡数据集的指标。
    """
    print("\n--- 4. 开始评估模型性能 ---")

    target_names = [f'Normal ({config.NORMAL_LABEL_VALUE})', f'Anomaly ({config.ANOMALY_LABEL_VALUE})']

    # 1. 打印基础指标
    accuracy = accuracy_score(y_true, y_pred)
    print(f"整体准确率 (Accuracy): {accuracy:.4f}  <-- 在不平衡数据中可能具有误导性！")

    # 2. 打印新增的、更可靠的指标
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}  <-- 更可靠的综合指标")

    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"马修斯相关系数 (MCC): {mcc:.4f}  <-- 衡量不平衡分类性能的优秀指标")

    # 3. 打印详细的分类报告 (依然非常重要)
    print("\n详细分类报告 (Classification Report):")
    # 关注 Anomaly 类别的 Precision, Recall, F1-score
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        labels=[config.NORMAL_LABEL_VALUE, config.ANOMALY_LABEL_VALUE]
    )
    print(report)

    # 4. 可视化混淆矩阵
    print("生成混淆矩阵可视化图...")
    cm = confusion_matrix(y_true, y_pred, labels=[config.NORMAL_LABEL_VALUE, config.ANOMALY_LABEL_VALUE])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names, yticklabels=target_names
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.show()


def evaluate_with_auprc(y_true, y_scores):
    """
    使用异常分数来计算并可视化Precision-Recall曲线，并计算AUPRC。
    """
    print("\n--- 4a. 开始进行 AUPRC 评估 ---")

    # 1. 计算PR曲线的点
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=config.ANOMALY_LABEL_VALUE)

    # 2. 计算PR曲线下面积 (AUPRC)
    auprc = auc(recall, precision)
    print(f"精确率-召回率曲线下面积 (AUPRC): {auprc:.4f}  <-- 这是最关键的指标！")

    # 3. 绘制PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'AUPRC = {auprc:.4f}')
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.xlabel('Recall (查全率)', fontsize=12)
    plt.ylabel('Precision (精确率)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
# ... (if __name__ == '__main__': 部分可以保持不变或删除) ...