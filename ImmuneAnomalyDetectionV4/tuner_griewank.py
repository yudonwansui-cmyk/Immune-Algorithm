# tuner_griewank.py

import numpy as np
import config_clonal as config
from immune_algorithm_clonal import ClonalSelectionAlgorithm
from test_functions import griewank_function


def tune_for_griewank():
    print("\n" + "=" * 50)
    print("=== 开始为 Griewank Function 寻找最优参数 ===")
    print("=" * 50)

    # --- 定义搜索空间 ---
    # 更大的种群可能有助于跳出局部最优
    pop_size_space = [50, 100, 150]
    # 较小的变异率有助于精细搜索
    mutation_rate_space = [0.01, 0.05, 0.1, 0.2]

    best_params = {}
    best_fitness = np.inf

    for pop_size in pop_size_space:
        for mutation_rate in mutation_rate_space:
            print(f"\n--- 测试参数: Population Size = {pop_size}, Mutation Rate = {mutation_rate} ---")

            model = ClonalSelectionAlgorithm(
                func=griewank_function,
                dim=config.FUNC_DIMENSION,
                bounds=[-600, 600],  # Griewank 的边界
                pop_size=pop_size,
                max_gen=config.MAX_GENERATIONS,
                clone_scale=config.CLONE_SCALE,
                mutation_rate=mutation_rate,
                num_replace=config.NUM_REPLACE
            )
            _, current_fitness = model.solve()

            print(f"结果 -> 最小值: {current_fitness}")

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_params = {'pop_size': pop_size, 'mutation_rate': mutation_rate}
                print(f"!!! 发现新的最优参数组合 !!!")

    print("\n" + "=" * 50)
    print("=== Griewank Function 调优完成 ===")
    print(f"找到的最优参数: {best_params}")
    print(f"对应的最低函数值: {best_fitness}")
    print("=" * 50)


if __name__ == '__main__':
    tune_for_griewank()