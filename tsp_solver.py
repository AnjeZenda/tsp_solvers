import numpy as np
import random
import math
from copy import deepcopy
from typing import Tuple, List, Dict
from python_tsp.exact.branch_and_bound import Node, PriorityQueue
from functools import lru_cache


class TSPSolver:
    def __init__(self, weight_matrix: np.ndarray):
        """Инициализация решателя TSP с матрицей весов"""
        self.weight_matrix = weight_matrix
        self.n = len(weight_matrix)  # Количество городов
        
    def solve(self, method: str, args: Dict = None) -> List[int]:
        """Общий интерфейс для всех алгоритмов"""
        if args is None:
            args = {}
            
        method = method.lower()
        if method == 'lin_kernighan':
            return self.lin_kernighan(**args)
        elif method == 'dynamic':
            return self.dynamic_programming(**args)
        elif method == 'branch_and_bound':
            return self.branch_and_bound(**args)
        elif method == 'simulated_annealing':
            return self.simulated_annealing(**args)
        elif method == 'genetic':
            return self.genetic_algorithm(**args)
        elif method == 'ant_colony':
            return self.ant_colony_optimization(**args)
        elif method == 'hybrid':
            return self.hybrid_algorithm(**args)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def lin_kernighan(self, **kwargs) -> List[int]:
        """Реализация алгоритма Лина-Кернигана"""
        max_iter = kwargs.get('max_iter', 200)
        max_k = kwargs.get('max_k', 3)
        
        tour = kwargs.get('tour', self._greedy_tour())
        best_length = self._tour_length(tour)
        
        for _ in range(max_iter):
            improved = False
            for k in range(2, max_k + 1):
                new_tour, new_length = self._k_opt_move(tour, k)
                if new_length < best_length:
                    tour, best_length = new_tour, new_length
                    improved = True
                    break
            if not improved:
                break
                
        return tour
    
    def _greedy_tour(self) -> List[int]:
        """Жадный алгоритм для начального решения"""
        unvisited = set(range(self.n))
        tour = [0]
        unvisited.remove(0)
        
        while unvisited:
            last = tour[-1]
            next_node = min(unvisited, key=lambda x: self.weight_matrix[last][x])
            tour.append(next_node)
            unvisited.remove(next_node)
            
        return tour
    
    def _k_opt_move(self, tour: List[int], k: int) -> Tuple[List[int], float]:
        """k-opt move для алгоритма Лина-Кернигана"""
        n = len(tour)
        best_tour, best_length = tour, self._tour_length(tour)
        
        for i in range(n):
            for j in range(i + 1, min(i + k, n)):
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_length = self._tour_length(new_tour)
                if new_length < best_length:
                    best_tour, best_length = new_tour, new_length
                    
        return best_tour, best_length
    
    def dynamic_programming(self, **kwargs) -> List[int]:
        N = frozenset(range(1, self.distance_matrix.shape[0]))
        memo: Dict[Tuple, int] = {}

        @lru_cache()
        def dist(ni: int, N: frozenset) -> float:
            if not N:
                return self.distance_matrix[ni, 0]

            costs = [
                (nj, self.distance_matrix[ni, nj] + dist(nj, N.difference({nj})))
                for nj in N
            ]
            nmin, min_cost = min(costs, key=lambda x: x[1])
            memo[(ni, N)] = nmin

            return min_cost

        best_distance = dist(0, N)


        ni = 0 
        solution = [0]

        while N:
            ni = memo[(ni, N)]
            solution.append(ni)
            N = N.difference({ni})

        return solution, best_distance
    
    def branch_and_bound(self, **kwargs) -> List[int]:
        num_cities = len(self.distance_matrix)
        cost_matrix = np.copy(self.distance_matrix).astype(float)
        np.fill_diagonal(cost_matrix, float('inf'))

        root = Node.from_cost_matrix(cost_matrix=cost_matrix)
        pq = PriorityQueue([root])

        while not pq.empty:
            min_node = pq.pop()

            if min_node.level == num_cities - 1:
                return min_node.path

            for index in range(num_cities):
                is_live_node = min_node.cost_matrix[min_node.index][index] != float('inf')
                if is_live_node:
                    live_node = Node.from_parent(parent=min_node, index=index)
                    pq.push(live_node)

        return []
    
    def _calculate_initial_lower_bound(self) -> float:
        """Вычисление начальной нижней границы (1-дерево)"""

        min_edges = []
        for i in range(self.n):
            edges = sorted([self.weight_matrix[i][j] for j in range(self.n) if j != i])
            min_edges.append(edges[0] + edges[1])
        
        return math.ceil(sum(min_edges) / 2)
    
    def _calculate_lower_bound(self, path: List[int], visited: set, prev_lb: float) -> float:
        """Вычисление нижней границы для частичного пути"""
        path_length = sum(self.weight_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        
        remaining = set(range(self.n)) - visited
        min_edges = 0
        for city in remaining:
            edges = []
            min_to_visited = min(self.weight_matrix[city][v] for v in visited)
            edges.append(min_to_visited)
            
            if len(remaining) > 1:
                min_to_remaining = min(self.weight_matrix[city][r] for r in remaining if r != city)
                edges.append(min_to_remaining)
            else:
                edges.append(self.weight_matrix[city][0])
            
            edges.sort()
            min_edges += edges[0] + edges[1]
        
        lb = path_length + math.ceil(min_edges / 2)
        
        if len(visited) == self.n:
            lb += self.weight_matrix[path[-1]][0]
            
        return lb
    
    def simulated_annealing(self, **kwargs) -> List[int]:
        """Метод отжига для приближенного решения"""
        temp = kwargs.get('initial_temp', 10000)
        cooling_rate = kwargs.get('cooling_rate', 0.002)
        max_iter = kwargs.get('max_iter', 10000)
        
        current_tour = kwargs.get('tour', self._greedy_tour())
        current_length = self._tour_length(current_tour)
        
        best_tour = current_tour.copy()
        best_length = current_length
        
        for i in range(max_iter):
            new_tour = self._get_2opt_neighbor(current_tour)
            new_length = self._tour_length(new_tour)
            
            if new_length < current_length or random.random() < math.exp((current_length - new_length) / temp):
                current_tour, current_length = new_tour, new_length
                
                if current_length < best_length:
                    best_tour, best_length = current_tour, current_length
            
            temp *= 1 - cooling_rate
            if temp < 1e-10:
                break
                
        return best_tour
    
    def _get_2opt_neighbor(self, tour: List[int]) -> List[int]:
        """Генерация соседнего решения с помощью 2-opt обмена"""
        n = len(tour)
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i
        return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    
    def genetic_algorithm(self, **kwargs) -> List[int]:
        """Генетический алгоритм для приближенного решения"""
        pop_size = kwargs.get('pop_size', 100)
        elite_size = kwargs.get('elite_size', 20)
        mutation_rate = kwargs.get('mutation_rate', 0.01)
        generations = kwargs.get('generations', 500)
        early_stop = kwargs.get('early_stop', 50)  
        
        population = [self._create_random_tour() for _ in range(pop_size)]
        
        best_tour = None
        best_length = float('inf')
        no_improvement = 0
        
        for generation in range(generations):
            fitness = [(self._tour_length(t), t) for t in population]
            ranked = sorted(fitness, key=lambda x: x[0])
            
            current_best_length, current_best_tour = ranked[0]
            
            if current_best_length < best_length:
                best_tour, best_length = current_best_tour, current_best_length
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= early_stop:
                break
            
            elite = [t for (_, t) in ranked[:elite_size]]
            
            children = elite.copy()
            
            while len(children) < pop_size:
                parent1, parent2 = random.sample(elite, 2)
                child = self._crossover(parent1, parent2)
                children.append(child)
            
            for i in range(len(children)):
                if random.random() < mutation_rate:
                    if random.random() < 0.5:
                        children[i] = self._mutate_swap(children[i])
                    else:
                        children[i] = self._mutate_invert(children[i])
            
            population = children
        
        return best_tour

    def _mutate_swap(self, tour: List[int]) -> List[int]:
        """Мутация обменом двух городов"""
        i, j = random.sample(range(self.n), 2)
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def _mutate_invert(self, tour: List[int]) -> List[int]:
        """Мутация инверсией сегмента"""
        i, j = sorted(random.sample(range(self.n), 2))
        tour[i:j+1] = tour[i:j+1][::-1]
        return tour
    
    def _create_random_tour(self) -> List[int]:
        """Создает случайный тур"""
        tour = list(range(self.n))
        random.shuffle(tour)
        return tour
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Упорядоченный кроссовер (OX)"""
        n = self.n
        start, end = sorted(random.sample(range(n), 2))
        
        child = [None] * n
        child[start:end] = parent1[start:end]
        
        ptr = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                if ptr >= n:
                    ptr = 0
                child[ptr] = city
                ptr += 1
                
        return child
    
    
    def ant_colony_optimization(self, **kwargs) -> List[int]:
        """Оптимизация муравьиной колонии для приближенного решения"""
        n_ants = kwargs.get('n_ants', 10)
        evaporation = kwargs.get('evaporation', 0.5)
        alpha = kwargs.get('alpha', 1)
        beta = kwargs.get('beta', 2)
        iterations = kwargs.get('iterations', 100)
        
        pheromone = np.ones((self.n, self.n)) / self.n
        
        best_tour = None
        best_length = float('inf')
        
        for _ in range(iterations):
            tours = []
            for _ in range(n_ants):
                tour = self._ant_tour(pheromone, alpha, beta)
                length = self._tour_length(tour)
                tours.append((tour, length))
                
                if length < best_length:
                    best_tour, best_length = tour, length
            
            pheromone *= (1 - evaporation)
            
            for tour, length in tours:
                for i in range(self.n):
                    j = (i + 1) % self.n
                    pheromone[tour[i]][tour[j]] += 1 / length
        
        return best_tour
    
    def _ant_tour(self, pheromone: np.ndarray, alpha: float, beta: float) -> List[int]:
        """Построение маршрута одним муравьем"""
        tour = [random.randint(0, self.n - 1)]
        unvisited = set(range(self.n)) - {tour[0]}
        
        while unvisited:
            last = tour[-1]
            probs = []
            total = 0
            for city in unvisited:
                p = (pheromone[last][city] ** alpha) * ((1 / self.weight_matrix[last][city]) ** beta)
                probs.append((city, p))
                total += p
            
            r = random.uniform(0, total)
            upto = 0
            for city, p in probs:
                if upto + p >= r:
                    tour.append(city)
                    unvisited.remove(city)
                    break
                upto += p
        
        return tour
    
    def _tour_length(self, tour: List[int]) -> float:
        """Вычисляет длину тура"""
        return sum(self.weight_matrix[tour[i]][tour[(i+1) % self.n]] for i in range(self.n))

    def hybrid_algorithm(self, **kwargs) -> List[int]:
        """Гибридный алгоритм"""
        iterations = kwargs.get('iter', 5)
        best_result = self.lin_kernighan(*kwargs)
        kwargs.update({'tour': best_result})
        for k in range(iterations):
            new_tour_sa = self.simulated_annealing(**kwargs)
            kwargs['tour'] = new_tour_sa
            new_tour_lk = self.lin_kernighan(**kwargs)
            kwargs['tour'] = new_tour_lk
            best_result = min(best_result, new_tour_sa, new_tour_lk, key=self._tour_length)
        return best_result