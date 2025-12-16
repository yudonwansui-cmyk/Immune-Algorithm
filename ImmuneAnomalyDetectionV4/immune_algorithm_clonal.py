# immune_algorithm_clonal.py

import numpy as np
from tqdm import tqdm


class ClonalSelectionAlgorithm:
    def __init__(self, func, dim, bounds, pop_size, max_gen, clone_scale, mutation_rate, num_replace):
        self.target_func = func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_generations = max_gen
        self.clone_scale = clone_scale
        self.mutation_rate = mutation_rate
        self.num_replace = num_replace

        # 初始化种群：在给定范围内随机生成抗体（解）
        self.population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.fitness_history = []

    def _calculate_fitness(self, solution):
        # 适应度就是目标函数的值，值越小越好
        return self.target_func(solution)

    def solve(self):
        for gen in tqdm(range(self.max_generations), desc="进化中"):
            # 1. 计算当前种群所有抗体的亲和度（适应度）
            fitness_values = np.apply_along_axis(self._calculate_fitness, 1, self.population)

            # 2. 根据亲和度进行排序（升序，因为我们要找最小值）
            sorted_indices = np.argsort(fitness_values)
            self.population = self.population[sorted_indices, :]
            fitness_values = fitness_values[sorted_indices]

            # 更新全局最优解
            if fitness_values[0] < self.best_fitness:
                self.best_fitness = fitness_values[0]
                self.best_solution = self.population[0, :].copy()
            self.fitness_history.append(self.best_fitness)

            # 3. 克隆操作
            clones = []
            for i in range(self.pop_size):
                # 亲和度越高（排名越靠前），克隆数量越多
                num_clones = int(self.clone_scale * (self.pop_size - i) / self.pop_size)
                for _ in range(num_clones):
                    clones.append(self.population[i, :].copy())
            clones = np.array(clones)

            # 4. 超变异操作
            for i in range(len(clones)):
                # 找到该克隆对应的原始抗体的适应度
                original_fitness = self._calculate_fitness(clones[i])
                # 适应度越高（值越小），变异率越低
                mutation_factor = np.exp(-self.mutation_rate * original_fitness)

                # 对克隆体进行高斯变异（小扰动）
                clone = clones[i]
                noise = np.random.normal(0, mutation_factor, self.dim)
                mutated_clone = clone + noise

                # 确保变异后的解仍在边界内
                mutated_clone = np.clip(mutated_clone, self.bounds[0], self.bounds[1])
                clones[i] = mutated_clone

            # 5. 种群更新/选择
            # 计算所有变异后克隆体的适应度
            clone_fitness = np.apply_along_axis(self._calculate_fitness, 1, clones)

            # 将原种群和变异克隆体合并
            combined_population = np.vstack([self.population, clones])
            combined_fitness = np.concatenate([fitness_values, clone_fitness])

            # 从合并后的种群中选出最好的 pop_size 个作为新一代
            sorted_indices_new = np.argsort(combined_fitness)
            self.population = combined_population[sorted_indices_new[:self.pop_size], :]

        print("优化完成！")
        return self.best_solution, self.best_fitness