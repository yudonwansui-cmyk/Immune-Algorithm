# data_processor.py (适配新版 '诊断' 数据集)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import config


def load_and_prepare_data():
    """
    负责加载、预处理和分割新版威斯康星乳腺癌数据集。
    """
    print("--- 1. 开始加载和处理 '诊断' 数据集 ---")

    # 1. 加载数据
    try:
        # 这个数据集第一行就是列名，所以可以直接读取
        df = pd.read_csv(config.DATA_PATH)
        print(f"成功加载数据集: {config.DATA_PATH}, 形状: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 数据文件未找到，请确保 '{config.DATA_PATH}' 路径正确。")
        return None, None, None

    # --- 2. 数据预处理 ---
    # a) 丢弃无关列
    # 根据截图，'id' 列和最后一列 'Unnamed: 32'（通常是空的）是无用的。
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['id', 'Unnamed: 32'], axis=1)
    else:
        df = df.drop('id', axis=1)

    # b) 标签编码：将 'M' 和 'B' 转换为数字
    # 这是非常关键的一步，因为后续的算法和评估函数通常处理数字标签。
    # 我们将 'B' (正常) 映射为 0，'M' (异常) 映射为 1。
    # 我们直接修改config里的值，让后续模块统一使用数字标签
    config.NORMAL_LABEL_VALUE = 0
    config.ANOMALY_LABEL_VALUE = 1
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].map(
        {'B': config.NORMAL_LABEL_VALUE, 'M': config.ANOMALY_LABEL_VALUE})
    print("标签已从 'B'/'M' 转换为 0/1。")

    # 3. 分离特征 (X) 和标签 (y)
    y = df[config.LABEL_COLUMN]
    X = df.drop(config.LABEL_COLUMN, axis=1)  # 除了标签列，其他都是特征

    # --- 4. 数据归一化 ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("所有30个特征数据已完成Min-Max归一化。")

    # --- 5. 数据集分割 ---
    # a) 提取所有正常样本用于训练
    normal_indices = (y == config.NORMAL_LABEL_VALUE)
    X_train_normal = X_scaled[normal_indices]

    # b) 划分测试集
    _, X_test, _, y_test = train_test_split(
        X_scaled, y.values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    print("\n--- 数据准备完成 ---")
    print(f"用于训练的正常样本 ('Self' set) 数量: {X_train_normal.shape[0]}")
    print(f"用于测试的样本总数: {X_test.shape[0]}")
    print(f"测试集中正常(0)样本数量: {np.sum(y_test == config.NORMAL_LABEL_VALUE)}")
    print(f"测试集中异常(1)样本数量: {np.sum(y_test == config.ANOMALY_LABEL_VALUE)}")
    print("-" * 20)

    return X_train_normal, X_test, y_test


if __name__ == '__main__':
    load_and_prepare_data()