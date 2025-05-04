import numpy as np
import random
import math
from itertools import permutations
from copy import deepcopy
from typing import Tuple, List, Dict, Callable
import heapq
from python_tsp.exact import solve_tsp_branch_and_bound, solve_tsp_dynamic_programming

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
        # Параметры по умолчанию
        max_iter = kwargs.get('max_iter', 100)
        max_k = kwargs.get('max_k', 5)
        
        # Начальное решение (жадный алгоритм)
        tour = kwargs.get('tour', self._greedy_tour())
        best_length = self._tour_length(tour)
        
        for _ in range(max_iter):
            improved = False
            for k in range(2, max_k + 1):
                # Генерируем k-оптимальные перестановки
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
                # Пробуем различные перестановки
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_length = self._tour_length(new_tour)
                if new_length < best_length:
                    best_tour, best_length = new_tour, new_length
                    
        return best_tour, best_length
    
    def dynamic_programming(self, **kwargs) -> List[int]:
        return solve_tsp_dynamic_programming(self.weight_matrix)[0]
        # """Точное решение методом динамического программирования (Хелд-Карп)"""
        # # Кэш для хранения подзадач: (последний_город, множество_городов) -> (длина, предыдущий_город)
        # memo = {}
        
        # # Инициализация: расстояние от стартового города (0) до каждого другого города
        # for i in range(1, self.n):
        #     memo[(i, frozenset([i]))] = (self.weight_matrix[0][i], 0)
        
        # # Заполняем таблицу для подмножеств размером от 2 до n-1
        # for subset_size in range(2, self.n):
        #     for subset in permutations([i for i in range(1, self.n)], subset_size):
        #         subset = frozenset(subset)
        #         for last in subset:
        #             # Минимизируем расстояние до подмножества subset \ {last}
        #             min_dist = float('inf')
        #             prev = -1
        #             for city in subset:
        #                 if city != last:
        #                     key = (city, subset - {last})
        #                     if key in memo:
        #                         dist = memo[key][0] + self.weight_matrix[city][last]
        #                         if dist < min_dist:
        #                             min_dist = dist
        #                             prev = city
        #             if prev != -1:
        #                 memo[(last, subset)] = (min_dist, prev)
        
        # # Находим минимальный тур, возвращающийся в город 0
        # full_subset = frozenset(range(1, self.n))
        # min_dist = float('inf')
        # last = -1
        # for city in range(1, self.n):
        #     key = (city, full_subset)
        #     if key in memo:
        #         dist = memo[key][0] + self.weight_matrix[city][0]
        #         if dist < min_dist:
        #             min_dist = dist
        #             last = city
        
        # # Восстанавливаем путь
        # if last == -1:
        #     return list(range(self.n))
        
        # tour = [0]
        # subset = full_subset
        # while subset:
        #     tour.append(last)
        #     new_subset = subset - {last}
        #     if not new_subset:
        #         break
        #     last = memo[(last, subset)][1]
        #     subset = new_subset
        # tour.append(0)
        
        # return tour[:-1]  # Убираем дублирование стартового города
    
    def branch_and_bound(self, **kwargs) -> List[int]:
        return solve_tsp_branch_and_bound(self.weight_matrix)[0]
        # """Метод ветвей и границ для точного решения"""
        # # Начальная верхняя граница (жадный алгоритм)
        # best_tour = self._greedy_tour()
        # best_length = self._tour_length(best_tour)
        
        # # Приоритетная очередь для узлов: (нижняя_граница, путь)
        # heap = []
        # initial_lower_bound = self._calculate_initial_lower_bound()
        # heapq.heappush(heap, (initial_lower_bound, [0], set([0])))
        
        # while heap:
        #     lower_bound, path, visited = heapq.heappop(heap)
            
        #     # Если текущая нижняя граница хуже лучшего решения, отбрасываем ветвь
        #     if lower_bound >= best_length:
        #         continue
                
        #     # Если все города посещены, завершаем тур
        #     if len(path) == self.n:
        #         complete_tour = path + [0]
        #         tour_length = self._tour_length(complete_tour)
        #         if tour_length < best_length:
        #             best_tour = complete_tour
        #             best_length = tour_length
        #         continue
                
        #     # Разветвление
        #     last = path[-1]
        #     for city in range(self.n):
        #         if city not in visited:
        #             new_path = path + [city]
        #             new_visited = set(visited)
        #             new_visited.add(city)
                    
        #             # Вычисляем новую нижнюю границу
        #             new_lower_bound = self._calculate_lower_bound(new_path, new_visited, lower_bound)
                    
        #             if new_lower_bound < best_length:
        #                 heapq.heappush(heap, (new_lower_bound, new_path, new_visited))
        
        # return best_tour[:-1]  # Убираем дублирование стартового города
    
    def _calculate_initial_lower_bound(self) -> float:
        """Вычисление начальной нижней границы (1-дерево)"""
        # Минимальные два ребра для каждого города
        min_edges = []
        for i in range(self.n):
            edges = sorted([self.weight_matrix[i][j] for j in range(self.n) if j != i])
            min_edges.append(edges[0] + edges[1])
        
        # Сумма минимальных ребер / 2 (округление вверх)
        return math.ceil(sum(min_edges) / 2)
    
    def _calculate_lower_bound(self, path: List[int], visited: set, prev_lb: float) -> float:
        """Вычисление нижней границы для частичного пути"""
        # Учитываем уже пройденные ребра
        path_length = sum(self.weight_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        
        # Для непосещенных городов учитываем два минимальных ребра
        remaining = set(range(self.n)) - visited
        min_edges = 0
        for city in remaining:
            edges = []
            # Минимальное ребро в/из посещенных городов
            min_to_visited = min(self.weight_matrix[city][v] for v in visited)
            edges.append(min_to_visited)
            
            # Минимальное ребро в/из непосещенных городов (кроме себя)
            if len(remaining) > 1:
                min_to_remaining = min(self.weight_matrix[city][r] for r in remaining if r != city)
                edges.append(min_to_remaining)
            else:
                # Если остался один город, учитываем ребро в начальный город
                edges.append(self.weight_matrix[city][0])
            
            edges.sort()
            min_edges += edges[0] + edges[1]
        
        # Общая нижняя граница
        lb = path_length + math.ceil(min_edges / 2)
        
        # Также учитываем ребро из последнего города обратно в начальный
        if len(visited) == self.n:
            lb += self.weight_matrix[path[-1]][0]
            
        return lb
    
    def simulated_annealing(self, **kwargs) -> List[int]:
        """Метод отжига для приближенного решения"""
        # Параметры алгоритма
        temp = kwargs.get('initial_temp', 10000)
        cooling_rate = kwargs.get('cooling_rate', 0.003)
        max_iter = kwargs.get('max_iter', 10000)
        
        # Начальное решение
        current_tour = kwargs.get('tour', self._greedy_tour())
        current_length = self._tour_length(current_tour)
        
        best_tour = current_tour.copy()
        best_length = current_length
        
        for i in range(max_iter):
            # Генерируем соседнее решение (2-opt обмен)
            new_tour = self._get_2opt_neighbor(current_tour)
            new_length = self._tour_length(new_tour)
            
            # Принимаем ли новое решение?
            if new_length < current_length or random.random() < math.exp((current_length - new_length) / temp):
                current_tour, current_length = new_tour, new_length
                
                # Обновляем лучшее решение
                if current_length < best_length:
                    best_tour, best_length = current_tour, current_length
            
            # Охлаждение
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
        # Параметры алгоритма
        pop_size = kwargs.get('pop_size', 100)
        elite_size = kwargs.get('elite_size', 20)
        mutation_rate = kwargs.get('mutation_rate', 0.01)
        generations = kwargs.get('generations', 500)
        early_stop = kwargs.get('early_stop', 50)  # Количество поколений без улучшений для ранней остановки
        
        # Инициализация популяции
        population = [self._create_random_tour() for _ in range(pop_size)]
        
        best_tour = None
        best_length = float('inf')
        no_improvement = 0
        
        for generation in range(generations):
            # Оценка приспособленности
            fitness = [(self._tour_length(t), t) for t in population]
            ranked = sorted(fitness, key=lambda x: x[0])
            
            current_best_length, current_best_tour = ranked[0]
            
            # Обновляем лучшее решение
            if current_best_length < best_length:
                best_tour, best_length = current_best_tour, current_best_length
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Ранняя остановка, если нет улучшений
            if no_improvement >= early_stop:
                break
            
            # Отбор элитных особей
            elite = [t for (_, t) in ranked[:elite_size]]
            
            # Формирование нового поколения
            children = elite.copy()
            
            # Кроссовер (упорядоченный)
            while len(children) < pop_size:
                parent1, parent2 = random.sample(elite, 2)
                child = self._crossover(parent1, parent2)
                children.append(child)
            
            # Мутация (обмен двух городов или инверсия сегмента)
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
        # Параметры алгоритма
        n_ants = kwargs.get('n_ants', 10)
        evaporation = kwargs.get('evaporation', 0.5)
        alpha = kwargs.get('alpha', 1)
        beta = kwargs.get('beta', 2)
        iterations = kwargs.get('iterations', 100)
        
        # Инициализация феромонов
        pheromone = np.ones((self.n, self.n)) / self.n
        
        best_tour = None
        best_length = float('inf')
        
        for _ in range(iterations):
            # Каждый муравей строит маршрут
            tours = []
            for _ in range(n_ants):
                tour = self._ant_tour(pheromone, alpha, beta)
                length = self._tour_length(tour)
                tours.append((tour, length))
                
                # Обновляем лучшее решение
                if length < best_length:
                    best_tour, best_length = tour, length
            
            # Испарение феромонов
            pheromone *= (1 - evaporation)
            
            # Обновление феромонов на лучших маршрутах
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
            # Вычисляем вероятности для каждого непосещенного города
            probs = []
            total = 0
            for city in unvisited:
                p = (pheromone[last][city] ** alpha) * ((1 / self.weight_matrix[last][city]) ** beta)
                probs.append((city, p))
                total += p
            
            # Выбираем следующий город
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
        iterations = kwargs.get('iter', 3)
        best_result = self.lin_kernighan(*kwargs)
        kwargs.update({'tour': best_result})
        for _ in range(iterations):
            new_tour_sa = self.simulated_annealing(**kwargs)
            kwargs['tour'] = new_tour_sa
            new_tour_lk = self.lin_kernighan(**kwargs)
            
            best_result = min(best_result, new_tour_sa, new_tour_lk, key=self._tour_length)
        
        return best_result