# config_clonal.py
from test_functions import rastrigin_function, sphere_function # 从函数库导入

# ==================================
# 1. 目标函数相关参数
# ==================================
# --- 在这里选择你要优化的函数 ---
TARGET_FUNCTION = rastrigin_function

# 定义函数维度
FUNC_DIMENSION = 2
# 定义每个变量的取值范围 [min, max]
VAR_BOUNDS = [-5.12, 5.12]

# ==================================
# 2. 克隆选择算法核心参数
# ==================================
POPULATION_SIZE = 100
MAX_GENERATIONS = 1000
CLONE_SCALE = 10
MUTATION_RATE = 0.1 # 这是一个很敏感的参数，需要调优
NUM_REPLACE = 5  # <--- 在这里添加这一行！！！