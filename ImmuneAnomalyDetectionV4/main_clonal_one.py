# main_clonal.py (单次实验最终版)

# 导入必要的模块和函数
import config_clonal as config
from immune_algorithm_clonal import ClonalSelectionAlgorithm
from visualization import plot_convergence_curve, plot_2d_function_contour
from test_functions import (rastrigin_function, sphere_function, ackley_function,
                            griewank_function, schwefel_function, rosenbrock_function)


def run_single_focused_experiment():
    """
    主函数，用于对 config_clonal.py 中指定的单个目标函数
    进行一次完整的、可重复的优化实验，并进行可视化展示。
    """

    # --- 步骤 0: 从config加载实验设置 ---

    # 创建一个函数信息库，用于查找全局最优解坐标
    function_info = {
        rastrigin_function: {'coords': [0.0, 0.0]},
        sphere_function: {'coords': [0.0, 0.0]},
        ackley_function: {'coords': [0.0, 0.0]},
        griewank_function: {'coords': [0.0, 0.0]},
        schwefel_function: {'coords': [420.9687, 420.9687]},
        rosenbrock_function: {'coords': [1.0, 1.0]}
    }

    selected_function = config.TARGET_FUNCTION
    selected_function_name = selected_function.__name__

    print("=" * 50)
    print(f"=== 开始单次优化实验: {selected_function_name} ===")
    print("=" * 50)

    # --- 步骤 1: (可选) 可视化目标函数 ---
    if config.FUNC_DIMENSION == 2:
        if selected_function in function_info:
            print("正在绘制目标函数的2D和3D图像...")
            min_coords = function_info[selected_function]['coords']
            plot_2d_function_contour(
                func=selected_function,
                bounds=config.VAR_BOUNDS,
                global_minimum_coords=min_coords,
                title=selected_function_name
            )
        else:
            print(f"警告: 函数 {selected_function_name} 的全局最小值信息未在 main_clonal.py 中定义。")

    # --- 步骤 2: 初始化算法 ---
    print("\n--- 1. 初始化算法 ---")
    print(f"参数配置:")
    print(f"  - Population Size: {config.POPULATION_SIZE}")
    print(f"  - Generations: {config.MAX_GENERATIONS}")
    print(f"  - Mutation Rate: {config.MUTATION_RATE}")

    model = ClonalSelectionAlgorithm(
        func=config.TARGET_FUNCTION,
        dim=config.FUNC_DIMENSION,
        bounds=config.VAR_BOUNDS,
        pop_size=config.POPULATION_SIZE,
        max_gen=config.MAX_GENERATIONS,
        clone_scale=config.CLONE_SCALE,
        mutation_rate=config.MUTATION_RATE,
        num_replace=config.NUM_REPLACE
    )

    # --- 步骤 3: 运行优化 ---
    print("\n--- 2. 开始运行优化 ---")
    best_solution, best_fitness = model.solve()

    # --- 步骤 4: 输出最终结果 ---
    print("\n--- 3. 优化完成 ---")
    print(f"找到的最优解坐标: {best_solution}")
    print(f"对应的函数最小值: {best_fitness:.8f}")  # 使用.8f格式化，显示更多小数位

    # --- 步骤 5: 可视化收敛过程 ---
    print("\n--- 4. 正在绘制收敛曲线 ---")
    plot_convergence_curve(model.fitness_history)

    print("\n" + "=" * 50)
    print("=== 实验流程执行完毕 ===")
    print("=" * 50)


if __name__ == '__main__':
    run_single_focused_experiment()