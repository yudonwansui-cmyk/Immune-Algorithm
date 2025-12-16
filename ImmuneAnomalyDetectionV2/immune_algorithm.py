# immune_algorithm.py

import numpy as np
from tqdm import tqdm  # 引入tqdm库来显示一个美观的进度条

import config  # 导入配置文件以获取算法参数


class NegativeSelectionAlgorithm:
    """
    实现基于负选择的人工免疫算法，用于异常检测。

    工作流程:
    1. `train` 方法: 使用正常数据（'self' samples）来生成一组不匹配任何正常数据的检测器。
    2. `predict` 方法: 使用生成的检测器来识别新数据，如果数据与任何检测器匹配，则被分类为异常（'non-self'）。
    """

    def __init__(self, radius, num_detectors, max_tries):
        """
        初始化负选择算法实例。

        Args:
            radius (float): 检测器的识别半径 (r)。这是算法最关键的超参数。
            num_detectors (int): 期望生成的成熟检测器的数量。
            max_tries (int): 生成单个检测器时，允许的最大尝试次数，防止死循环。
        """
        # --- 核心参数 ---
        self.radius = radius
        self.num_detectors_target = num_detectors

        # --- 保护性参数 ---
        self.max_tries_per_detector = max_tries

        # --- 存储结果 ---
        self.detectors = None  # 初始化检测器列表为空，训练后填充

    @staticmethod
    def _calculate_distance(point1, point2):
        """
        计算两个数据点之间的欧氏距离。
        这是一个静态方法，因为它不依赖于任何实例特定的状态（如self.radius）。

        Args:
            point1 (np.array): 第一个点的坐标。
            point2 (np.array): 第二个点的坐标。

        Returns:
            float: 两点间的欧氏距离。
        """
        # np.linalg.norm 是一个计算向量范数的函数，默认计算L2范数（即欧氏距离），非常高效。
        return np.linalg.norm(point1 - point2)

    def train(self, self_samples):
        """
        训练模型，即生成一组“成熟的”检测器。

        Args:
            self_samples (np.array): 一个二维numpy数组，每行代表一个归一化后的正常样本。
        """
        print("\n--- 2. 开始训练模型 (生成检测器) ---")

        if self_samples.shape[0] == 0:
            raise ValueError("用于训练的正常样本集不能为空！")

        n_features = self_samples.shape[1]
        generated_detectors = []

        # 使用tqdm创建一个进度条，total参数设定了进度条的目标长度
        pbar = tqdm(total=self.num_detectors_target, desc="生成检测器")

        # 循环直到生成足够数量的检测器
        while len(generated_detectors) < self.num_detectors_target:

            # --- 生成候选检测器 ---
            is_candidate_valid = False
            tries = 0
            candidate_detector = None

            # 尝试找到一个有效的候选检测器
            while not is_candidate_valid and tries < self.max_tries_per_detector:
                # 1. 在特征空间中随机生成一个候选检测器。
                #    因为数据已经归一化到[0, 1]，所以我们也在这个范围内生成。
                candidate_detector = np.random.rand(n_features)

                # 2. 检查该候选者是否与任何“自我”样本发生冲突（距离太近）。
                is_too_close_to_self = False
                for self_point in self_samples:
                    if self._calculate_distance(candidate_detector, self_point) < self.radius:
                        is_too_close_to_self = True
                        break  # 一旦发现冲突，立即停止检查此候选者，它是无效的。

                # 如果候选者没有与任何自我样本冲突，它就是一个有效的检测器
                if not is_too_close_to_self:
                    is_candidate_valid = True

                tries += 1

            # --- 存储有效的检测器 ---
            if is_candidate_valid:
                generated_detectors.append(candidate_detector)
                pbar.update(1)  # 进度条前进一步
            else:
                # 如果在最大尝试次数内都找不到一个有效的检测器，很可能是参数设置问题
                print(f"\n警告: 在尝试 {self.max_tries_per_detector} 次后仍无法生成新的有效检测器。")
                print(f"当前已生成 {len(generated_detectors)} / {self.num_detectors_target} 个。")
                print("可能的原因是 DETECTOR_RADIUS 过大，导致没有空间留给检测器。")
                print("训练提前终止。")
                break

        pbar.close()  # 关闭进度条

        self.detectors = np.array(generated_detectors)

        if self.detectors.shape[0] == 0:
            print("\n严重警告：未能生成任何有效的检测器！预测将全部判为正常。")
        else:
            print(f"--- 训练完成！最终成功生成 {self.detectors.shape[0]} 个检测器。---")

    def predict(self, test_samples):
        """
        使用已生成的检测器对新样本进行分类。

        Args:
            test_samples (np.array): 一个二维numpy数组，每行代表一个待检测的样本。

        Returns:
            np.array: 一个一维numpy数组，包含每个样本的预测标签
                      (config.ANOMALY_LABEL_VALUE 表示异常, config.NORMAL_LABEL_VALUE 表示正常)。
        """
        print("\n--- 3. 开始在测试集上进行预测 ---")

        if self.detectors is None or self.detectors.shape[0] == 0:
            print("警告: 模型未训练或没有有效的检测器。所有样本将被预测为正常。")
            # 返回一个全为“正常”标签的数组
            return np.full(test_samples.shape[0], config.NORMAL_LABEL_VALUE)

        predictions = []
        # 遍历每一个需要测试的样本
        for test_point in tqdm(test_samples, desc="预测中"):
            is_anomaly = False
            # 将测试样本与每一个成熟的检测器进行比较
            for detector in self.detectors:
                if self._calculate_distance(test_point, detector) < self.radius:
                    # 如果测试样本落入任何一个检测器的识别范围内，就立即判定为异常
                    is_anomaly = True
                    break  # 无需再与其他检测器比较

            # 根据是否被检测为异常，添加对应的标签值
            if is_anomaly:
                predictions.append(config.ANOMALY_LABEL_VALUE)
            else:
                predictions.append(config.NORMAL_LABEL_VALUE)

        print("--- 预测完成 ---")
        return np.array(predictions)

    def predict_scores(self, test_samples):
        """
        为每个测试样本计算一个“异常分数”。
        分数定义为： -1 * (到最近检测器的最小距离)
        分数越高（越接近0），代表越可能是异常。
        """
        print("\n--- 3a. 开始计算异常分数 ---")
        if self.detectors is None or self.detectors.shape[0] == 0:
            print("警告: 模型未训练，无法计算分数。")
            return np.full(test_samples.shape[0], -np.inf)  # 返回一个极小值

        scores = []
        for test_point in tqdm(test_samples, desc="计算分数中"):
            min_distance = np.inf
            for detector in self.detectors:
                distance = self._calculate_distance(test_point, detector)
                if distance < min_distance:
                    min_distance = distance

            # 分数 = -最小距离。这样距离越小，分数越大。
            scores.append(-min_distance)

        return np.array(scores)


# (可选) 用于独立测试该模块的简单示例
if __name__ == '__main__':
    # 创建一个简单的模拟数据集
    # 正常数据集中在左下角
    mock_self_samples = np.random.rand(100, 2) * 0.3
    # 测试数据包含一些正常和一些异常（在右上角）
    mock_test_normal = np.random.rand(10, 2) * 0.3
    mock_test_anomaly = (np.random.rand(10, 2) * 0.3) + 0.7
    mock_test_samples = np.vstack([mock_test_normal, mock_test_anomaly])

    print("--- 开始模块独立测试 ---")
    # 初始化算法
    # 使用config.py中定义的参数
    model = NegativeSelectionAlgorithm(
        radius=config.DETECTOR_RADIUS,
        num_detectors=config.NUM_DETECTORS_TO_GENERATE,
        max_tries=config.MAX_GENERATION_TRIES
    )

    # 训练
    model.train(mock_self_samples)

    # 预测
    if model.detectors is not None and model.detectors.shape[0] > 0:
        predictions = model.predict(mock_test_samples)

        # 打印结果
        print("\n--- 测试结果 ---")
        print(f"模拟的异常样本 (后10个) 的预测标签: {predictions[-10:]}")
        print(f"预期异常标签为: {config.ANOMALY_LABEL_VALUE}")

        num_correct_anomalies = np.sum(predictions[-10:] == config.ANOMALY_LABEL_VALUE)
        print(f"成功识别出 {num_correct_anomalies} / 10 个异常样本。")
