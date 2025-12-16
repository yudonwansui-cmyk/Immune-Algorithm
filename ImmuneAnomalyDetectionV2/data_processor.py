# data_processor.py (V3 - 适配信用卡欺诈检测数据集)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import config


def load_and_prepare_data():
    """
    负责加载、采样、预处理和分割信用卡欺诈数据集。
    """
    print("--- 1. 开始加载和处理 '信用卡欺诈' 数据集 ---")

    # 1. 加载数据
    try:
        df = pd.read_csv(config.DATA_PATH)
        print(f"成功加载完整数据集: {config.DATA_PATH}, 形状: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 数据文件未找到，请确保 '{config.DATA_PATH}' 路径正确。")
        return None, None, None

    # --- 2. 数据采样 (可选但强烈推荐) ---
    if config.USE_SAMPLING:
        print(f"启用采样模式...")
        df_normal = df[df[config.LABEL_COLUMN] == 0]
        df_anomaly = df[df[config.LABEL_COLUMN] == 1]

        df_normal_sampled = df_normal.sample(frac=config.NORMAL_SAMPLE_FRAC, random_state=config.RANDOM_STATE)
        df_anomaly_sampled = df_anomaly.sample(frac=config.ANOMALY_SAMPLE_FRAC, random_state=config.RANDOM_STATE)

        df = pd.concat([df_normal_sampled, df_anomaly_sampled])
        print(f"采样后数据集形状: {df.shape}")
        print(f"其中正常样本: {df_normal_sampled.shape[0]}, 异常样本: {df_anomaly_sampled.shape[0]}")

    # --- 3. 数据预处理与特征工程 ---
    # a) 分离特征 (X) 和标签 (y)
    y = df[config.LABEL_COLUMN]
    X = df.drop(config.LABEL_COLUMN, axis=1)

    # b) 特征归一化
    # 'Time' 和 'Amount' 的尺度与其他V1-V28特征差异巨大，必须进行归一化。
    # StandardScaler 更适合处理这种数据分布，它将数据转换为均值为0，方差为1。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("所有特征数据已完成StandardScaler归一化。")

    # --- 4. 数据集分割 ---
    # a) 提取所有正常样本用于训练
    normal_indices = (y == config.NORMAL_LABEL_VALUE)
    X_train_normal = X_scaled[normal_indices]

    # b) 划分测试集
    _, X_test, _, y_test = train_test_split(
        X_scaled, y.values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y  # stratify在这里至关重要，保证测试集中有稀有的欺诈样本
    )

    print("\n--- 数据准备完成 ---")
    print(f"用于训练的正常样本 ('Self' set) 数量: {X_train_normal.shape[0]}")
    print(f"用于测试的样本总数: {X_test.shape[0]}")
    print(f"测试集中正常({config.NORMAL_LABEL_VALUE})样本数量: {np.sum(y_test == config.NORMAL_LABEL_VALUE)}")
    print(f"测试集中异常({config.ANOMALY_LABEL_VALUE})样本数量: {np.sum(y_test == config.ANOMALY_LABEL_VALUE)}")
    print("-" * 20)

    return X_train_normal, X_test, y_test


if __name__ == '__main__':
    load_and_prepare_data()