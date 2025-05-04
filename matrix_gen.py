import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Генерация случайных чисел
np.random.seed(42)

### Функция 1: Генерация матрицы с кластерами вершин
def generate_cluster_matrix(k, m):
    """
    Генерирует матрицу задачи коммивояжера с кластерами вершин.
    
    :param k: Количество кластеров
    :param m: Количество вершин в каждом кластере
    :return: Симметричная матрица расстояний
    """
    n = k * m  # Общее количество городов
    matrix = np.zeros((n, n))
    
    # Генерируем координаты городов
    for cluster_id in range(k):
        cluster_coords = np.random.rand(m, 2) * 5  # Координаты внутри кластера
        for i in range(m):
            for j in range(m):
                if i != j:
                    # Расстояние между вершинами внутри кластера
                    dist = np.linalg.norm(cluster_coords[i] - cluster_coords[j])
                    matrix[cluster_id * m + i, cluster_id * m + j] = dist
                    matrix[cluster_id * m + j, cluster_id * m + i] = dist
    
    # Увеличиваем расстояния между вершинами разных кластеров
    for i in range(n):
        for j in range(i + 1, n):
            if i // m != j // m:  # Если вершины из разных кластеров
                matrix[i, j] = np.random.uniform(40, 60)  # Большое расстояние
                matrix[j, i] = matrix[i, j]
    
    for i in range(n):
        matrix[i, i] = float('inf')
    
    return matrix


def random_cities_points(r, cities, l):
    while len(cities) < l:
        x = np.random.randint(-r, r)
        y = np.random.randint(-r, r)
        if x*x + y*y <= r*r and (x, y) not in cities:
            cities.append((x, y))
    return cities

### Функция 2: Генерация матрицы с городами на окружности
def generate_circle_matrix(r, n):
    """
    Генерирует матрицу задачи коммивояжера с городами, равномерно расположенными на окружности.
    
    :param r: Радиус окружности
    :param n: Количество городов
    :return: Симметричная матрица расстояний
    """
    
    cities = random_cities_points(r, [], n)
    matrix = np.zeros((n, n))
    for i in range(len(cities) - 1):
        for j in range(i+1, len(cities)):
            dist = np.sqrt((cities[i][0] - cities[j][0]) ** 2 + (cities[i][1] - cities[j][1]) ** 2)
            matrix[i, j] = dist
            matrix[j, i] = dist
    for i in range(n):
        matrix[i, i] = float('inf')
    return matrix

### Функция 3: Генерация матрицы с радиальными кластерами
def generate_radial_clusters_matrix(n, num_clusters, k = 0, l = 1, h = 1000):
    """
    Генерирует матрицу задачи коммивояжера с радиальными кластерами.
    
    :param n: Количество городов
    :param num_clusters: Количество радиальных кластеров
    :return: Симметричная матрица расстояний
    """
    matrix = np.zeros((n, n))
    cluster_size = n // num_clusters
    S = [(l, h) if i == 0 else (l+np.power(k, i), h+np.power(k, i)) for i in range(num_clusters)]
    
    for i in range(n):
        for j in range(i+1, n):
            if i >= j: continue
            radius_number = max(i // cluster_size, j // cluster_size)
            current_range = S[radius_number]
            matrix[i, j] = np.random.randint(current_range[0], current_range[1], dtype="uint64")
            matrix[j, i] = matrix[i, j]
    for i in range(n):
        matrix[i, i] = float('inf')
    return matrix

def draw_graph_from_weight_matrix(weight_matrix, type=None):
    """
    Рисует граф по матрице весов.
    weight_matrix - квадратная матрица numpy или список списков,
                    где weight_matrix[i][j] - вес ребра из i в j.
                    Если вес 0 или None - ребро отсутствует.
    """
    G = nx.Graph()  # Ориентированный граф. Для неориентированного используйте nx.Graph()

    n = len(weight_matrix)
    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(n):
            w = weight_matrix[i][j]
            if w is not None and w != 0:
                if type == 'cluster' and w < 30:
                    G.add_edge(i, j, weight=w)
                elif type == 'radial':
                    G.add_edge(i, j, weight=w)
                else:
                    G.add_edge(i, j, weight=w)
                    

    pos = nx.spring_layout(G, seed=42)  # Расположение узлов
    # Рисуем узлы
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=20)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

    # Рисуем ребра
    nx.draw_networkx_edges(G, pos, arrowstyle='-', arrowsize=1, edge_color="white")


    plt.title("Граф по матрице весов")
    plt.axis('equal')
    plt.show()

### Главная функция для демонстрации
def start():
    # Тип 1: Матрица с кластерами вершин
    k = 3  # Количество кластеров
    m = 25  # Количество вершин в каждом кластере
    cluster_matrix = generate_cluster_matrix(k, m)
    print("Матрица с кластерами вершин:")
    print(cluster_matrix[:10, :10])  # Показываем часть матрицы
    draw_graph_from_weight_matrix(cluster_matrix, 'cluster')

    # Тип 2: Матрица с городами на окружности
    r = 100  # Радиус окружности
    n = 60   # Количество городов
    circle_matrix, coords = generate_circle_matrix(r, n)
    print("\nМатрица с городами на окружности:")
    print(circle_matrix[:5, :5])  # Показываем часть матрицы
    draw_graph_from_weight_matrix(circle_matrix)
    # Визуализация городов на окружности
    plt.figure(figsize=(6, 6))
    plt.scatter(list(map(lambda x: x[0], coords)), list(map(lambda x: x[1], coords)), s=3, color='blue',)
    circle = plt.Circle((0, 0), r, color='red', fill=False)
    plt.gca().add_patch(circle)
    plt.title("Города на окружности")
    plt.axis('equal')
    plt.show()

    # Тип 3: Матрица с радиальными кластерами
    n = 12  # Количество городов
    num_clusters = 4 # Количество радиальных кластеров
    radial_matrix = generate_radial_clusters_matrix(n, num_clusters)
    print("\nМатрица с радиальными кластерами:")
    print(radial_matrix[:25, :25])  # Показываем часть матрицы
    draw_graph_from_weight_matrix(radial_matrix, 'radial')

if __name__ == "__main__":
    start()