# tuner_sphere.py

import numpy as np
import config_clonal as config
from immune_algorithm_clonal import ClonalSelectionAlgorithm
from test_functions import sphere_function


def tune_for_sphere():
    print("\n" + "=" * 50)
    print("=== 开始为 Sphere Function 寻找最优参数 ===")
    print("=" * 50)

    # --- 定义搜索空间 ---
    # 种群大小可以不用太大
    pop_size_space = [30, 50, 80]
    # 变异率是关键，我们需要一个合适的范围来探索
    mutation_rate_space = [0.1, 0.5, 1.0, 1.5, 2.0]

    # 初始化记录变量
    best_params = {}
    best_fitness = np.inf

    # 嵌套循环搜索
    for pop_size in pop_size_space:
        for mutation_rate in mutation_rate_space:
            print(f"\n--- 测试参数: Population Size = {pop_size}, Mutation Rate = {mutation_rate} ---")

            # 使用当前参数组合进行一次完整的优化
            model = ClonalSelectionAlgorithm(
                func=sphere_function,
                dim=config.FUNC_DIMENSION,
                bounds=[-100, 100],  # Sphere 的边界
                pop_size=pop_size,
                max_gen=config.MAX_GENERATIONS,
                clone_scale=config.CLONE_SCALE,
                mutation_rate=mutation_rate,
                num_replace=config.NUM_REPLACE
            )
            _, current_fitness = model.solve()

            print(f"结果 -> 最小值: {current_fitness}")

            # 检查是否是新的最优解
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_params = {'pop_size': pop_size, 'mutation_rate': mutation_rate}
                print(f"!!! 发现新的最优参数组合 !!!")

    print("\n" + "=" * 50)
    print("=== Sphere Function 调优完成 ===")
    print(f"找到的最优参数: {best_params}")
    print(f"对应的最低函数值: {best_fitness}")
    print("=" * 50)


if __name__ == '__main__':
    tune_for_sphere()