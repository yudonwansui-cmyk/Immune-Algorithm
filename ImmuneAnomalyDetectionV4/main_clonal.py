# main_clonal.py (批量实验终极专业版)

# 导入必要的模块和函数
import config_clonal as config
from immune_algorithm_clonal import ClonalSelectionAlgorithm
from visualization import plot_convergence_curve, plot_2d_function_contour
from test_functions import (rastrigin_function, sphere_function, ackley_function,
                            griewank_function, schwefel_function, rosenbrock_function)


def run_single_experiment(target_func, func_config):
    """
    针对单个函数执行一次完整的优化实验。
    它会从传入的 func_config 字典中获取该函数的所有专属配置。
    """
    func_name = target_func.__name__

    # 打印实验分隔符和标题
    print("\n" + "=" * 60)
    print(f"=== 开始优化目标函数: {func_name} ===")
    print("=" * 60)

    # --- 步骤 0: 可视化目标函数 ---
    if config.FUNC_DIMENSION == 2 and func_config.get('coords'):
        print(f"正在绘制 {func_name} 的2D和3D图像...")
        plot_2d_function_contour(
            func=target_func,
            bounds=func_config['bounds'],
            global_minimum_coords=func_config['coords'],
            title=func_name
        )

    # --- 步骤 1: 初始化算法 ---
    print(f"\n--- 1. 初始化算法 (函数: {func_name}) ---")
    print("使用该函数的专属最优参数:")
    print(f"  - Population Size: {func_config['pop_size']}")
    print(f"  - Mutation Rate:   {func_config['mutation_rate']}")

    # 使用从 func_config 传入的专属参数和从 config.py 传入的全局参数来创建模型
    model = ClonalSelectionAlgorithm(
        func=target_func,
        dim=config.FUNC_DIMENSION,
        bounds=func_config['bounds'],
        pop_size=func_config['pop_size'],
        max_gen=config.MAX_GENERATIONS,
        clone_scale=config.CLONE_SCALE,
        mutation_rate=func_config['mutation_rate'],
        num_replace=config.NUM_REPLACE
    )

    # --- 步骤 2 & 3: 运行优化并输出结果 ---
    print(f"\n--- 2. 开始运行优化 (函数: {func_name}) ---")
    best_solution, best_fitness = model.solve()

    print(f"\n--- 3. {func_name} 优化完成 ---")
    print(f"找到的最优解坐标: {best_solution}")
    print(f"对应的函数最小值: {best_fitness:.8f}")

    # --- 步骤 4: 可视化收敛过程 ---
    print(f"\n--- 4. 正在绘制 {func_name} 的收敛曲线 ---")
    plot_convergence_curve(model.fitness_history)


def main():
    """
    主程序入口，定义所有实验的配置清单，并依次执行。
    """
    # !!! ========================== 关键配置区域 ========================== !!!
    # !!!                                                                    !!!
    # !!!  在这里为你通过 tuner_xxx.py 找到的每个函数的最优参数填入下方。  !!!
    # !!!                                                                    !!!
    # !!! ================================================================== !!!
    experiments_to_run = {
        "Rastrigin": {
            "function": rastrigin_function,
            "bounds": [-5.12, 5.12],
            "coords": [0.0, 0.0],
            "pop_size": 100,  # <--- 填入 Rastrigin 的最优种群大小
            "mutation_rate": 0.1  # <--- 填入 Rastrigin 的最优变异率
        },
        "Sphere": {
            "function": sphere_function,
            "bounds": [-100, 100],
            "coords": [0.0, 0.0],
            "pop_size": 50,  # <--- 填入 Sphere 的最优种群大小
            "mutation_rate": 1.5  # <--- 填入 Sphere 的最优变异率 (示例值)
        },
        "Ackley": {
            "function": ackley_function,
            "bounds": [-32.768, 32.768],
            "coords": [0.0, 0.0],
            "pop_size": 100,
            "mutation_rate": 0.1
        },
        "Griewank": {
            "function": griewank_function,
            "bounds": [-600, 600],
            "coords": [0.0, 0.0],
            "pop_size": 150,
            "mutation_rate": 0.05
        },
        "Schwefel": {
            "function": schwefel_function,
            "bounds": [-500, 500],
            "coords": [420.9687, 420.9687],
            "pop_size": 300,
            "mutation_rate": 0.5
        },
        "Rosenbrock": {
            "function": rosenbrock_function,
            "bounds": [-30, 30],
            "coords": [1.0, 1.0],
            "pop_size": 100,
            "mutation_rate": 0.1
        }
    }

    # 循环执行所有在上面定义的实验
    for name, info in experiments_to_run.items():
        run_single_experiment(info['function'], info)

    print("\n" + "=" * 60)
    print("=== 所有实验已执行完毕 ===")
    print("=" * 60)


if __name__ == '__main__':
    main()