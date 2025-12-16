# test_functions.py
import numpy as np

def rastrigin_function(solution):
    """
    Rastrigin 函数.
    全局最小值在 solution 为全0向量时取得, 值为0。
    """
    return np.sum(solution**2 - 10 * np.cos(2 * np.pi * solution) + 10)

def sphere_function(solution):
    """
    Sphere 函数 (最简单的测试函数).
    全局最小值在 solution 为全0向量时取得, 值为0。
    """
    return np.sum(solution**2)

def griewank_function(solution):
    """
    Griewank 函数.
    特点是强多峰，有一个被无数局部最优解包围的全局最优解。
    全局最小值在 solution 为全0向量时取得, 值为0。
    """
    sum_term = np.sum(solution ** 2 / 4000)
    prod_term = np.prod(np.cos(solution / np.sqrt(np.arange(1, len(solution) + 1))))
    return sum_term - prod_term + 1

def schwefel_function(solution):
    """
    Schwefel 函数.
    特点是具有欺骗性，次优解离全局最优解很远。
    全局最小值在 solution 为全 420.9687 向量时取得, 值为0。
    """
    n = len(solution)
    return 418.9829 * n - np.sum(solution * np.sin(np.sqrt(np.abs(solution))))

def rosenbrock_function(solution):
    """
    Rosenbrock 函数 (香蕉函数).
    特点是全局最优解在一个狭长、平缓、弯曲的山谷中。
    全局最小值在 solution 为全1向量时取得, 值为0。
    """
    return np.sum(100 * (solution[1:] - solution[:-1] ** 2) ** 2 + (1 - solution[:-1]) ** 2)




def ackley_function(solution):
    """
    Ackley 函数.
    全局最小值在 solution 为全0向量时取得, 值为0。
    """
    n = len(solution)
    sum1 = np.sum(solution**2)
    sum2 = np.sum(np.cos(2 * np.pi * solution))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.e