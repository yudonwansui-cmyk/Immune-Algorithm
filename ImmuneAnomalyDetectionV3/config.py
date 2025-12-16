# config.py (V-Wine - 适配红葡萄酒质量数据集)

# ==================================
# 1. 数据集相关参数
# ==================================
DATA_PATH = "data/winequality-red.csv"

# 特征列名列表（除了 'quality'）
# 我们可以在 data_processor.py 中自动获取，这里可以不写

# 目标/标签所在的列名
LABEL_COLUMN = 'quality'

# !!! 新增：定义异常的质量阈值 !!!
# 大于等于这个值的将被视为“异常”(Anomaly)
QUALITY_THRESHOLD = 7

# 定义最终的二分类标签值
NORMAL_LABEL_VALUE = 0  # 代表“普通酒”
ANOMALY_LABEL_VALUE = 1 # 代表“优质酒”


# ==================================
# 2. 数据处理相关参数
# ==================================
# 这个数据集不大，我们不再需要采样
USE_SAMPLING = False # 关闭采样

TEST_SIZE = 0.3
RANDOM_STATE = 42


# ==================================
# 3. 人工免疫算法核心参数
# ==================================
# 同样，需要为这个新数据集重新寻找最优参数
DETECTOR_RADIUS = 1.0  # 一个全新的起始猜测值
NUM_DETECTORS_TO_GENERATE = 1000
MAX_GENERATION_TRIES = 1000000