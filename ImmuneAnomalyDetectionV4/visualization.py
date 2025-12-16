import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # 用于改善视觉效果


def plot_convergence_curve(fitness_history):
    """绘制算法的收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', label='Best Fitness')
    plt.title("Convergence Curve of Clonal Selection Algorithm")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Function Value)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_2d_function_contour(func, bounds, global_minimum_coords, title="Function Contour Plot"):
    """
    绘制二维函数的等高线图和3D曲面图，并标记全局最小值。

    Args:
        func (function): 要可视化的函数 (例如, rastrigin_function)。
        bounds (list): 函数变量的范围, [min_val, max_val]。
        global_minimum_coords (list or np.array): 全局最小值的坐标, [x, y]。
        title (str): 图表标题。
    """
    # 1. 创建网格数据
    x = np.linspace(bounds[0], bounds[1], 400)
    y = np.linspace(bounds[0], bounds[1], 400)
    X, Y = np.meshgrid(x, y)

    # 2. 计算每个网格点的函数值
    # np.apply_along_axis 在高维数组上应用函数会很慢，我们手动处理
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # 3. 创建一个包含两个子图的画布
    fig = plt.figure(figsize=(18, 8))

    # --- 第一个子图: 2D 等高线图 ---
    ax1 = fig.add_subplot(1, 2, 1)
    # 使用对数范数可以让颜色变化更明显，更容易看出细节
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', norm=LogNorm())
    fig.colorbar(contour, ax=ax1, label='Function Value (log scale)')

    # 标记全局最小值点
    ax1.plot(global_minimum_coords[0], global_minimum_coords[1], 'r*', markersize=15, label='Global Minimum')

    ax1.set_title(f'2D Contour Plot of {title}', fontsize=16)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.legend()
    ax1.grid(True)

    # --- 第二个子图: 3D 曲面图 ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', norm=LogNorm())

    # 在3D图上标记全局最小值点
    min_val = func(np.array(global_minimum_coords))
    ax2.scatter(global_minimum_coords[0], global_minimum_coords[1], min_val, color='red', s=100, label='Global Minimum',
                depthshade=True)

    ax2.set_title(f'3D Surface Plot of {title}', fontsize=16)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('Function Value (log scale)')
    ax2.legend()

    plt.tight_layout()
    plt.show()
