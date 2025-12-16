# data_processor.py (V-Wine - 适配红葡萄酒质量数据集)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


def load_and_prepare_data():
    """
    负责加载、转换、预处理和分割红葡萄酒质量数据集。
    """
    print("--- 1. 开始加载和处理 '红葡萄酒质量' 数据集 ---")

    # 1. 加载数据
    try:
        # 这个数据集的列名分隔符是分号';'，需要特别指定
        df = pd.read_csv(config.DATA_PATH, sep=';')
        print(f"成功加载数据集: {config.DATA_PATH}, 形状: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 数据文件未找到，请确保 '{config.DATA_PATH}' 路径正确。")
        return None, None, None

    # --- 2. 关键步骤：将多分类问题转化为二分类异常检测问题 ---
    # 根据 config 中设定的阈值，创建新的二分类标签列
    df['binary_quality'] = df[config.LABEL_COLUMN].apply(
        lambda x: config.ANOMALY_LABEL_VALUE if x >= config.QUALITY_THRESHOLD else config.NORMAL_LABEL_VALUE
    )
    # 更新 config 中的标签列名，让后续步骤使用新的二分类标签
    config.LABEL_COLUMN = 'binary_quality'
    print(f"已将 'quality' 列转换为二分类标签： >= {config.QUALITY_THRESHOLD} 的为异常(1)。")

    # 打印转换后的类别分布
    print("转换后的类别分布:")
    print(df[config.LABEL_COLUMN].value_counts())

    # --- 3. 数据预处理 ---
    # a) 分离特征 (X) 和标签 (y)
    y = df[config.LABEL_COLUMN]
    # 丢弃原始的 quality 列和新的二分类标签列，剩下的都是特征
    X = df.drop(['quality', 'binary_quality'], axis=1)

    # b) 特征归一化
    # 所有特征都是数值型，可以直接进行归一化
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
        stratify=y
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