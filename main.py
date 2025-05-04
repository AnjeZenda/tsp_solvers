import numpy as np
from tsp_solver import TSPSolver
import timeit
from matrix_gen import *
import pprint
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def save_result(method, name, result, k):
    result[name][k] = {}
    result[name][k]['tour'] = method(name)

def experiment(ks, s1_range, count_cities, count_radius, methods):
    print("start experiment")
    results = {
        'lin_kernighan': {},
        'dynamic': {},
        'branch_and_bound': {},
        'simulated_annealing': {},
        'genetic': {},
        'ant_colony': {},
        'hybrid': {}
    }
    

    for k in ks:
        matrix = generate_radial_clusters_matrix(count_cities, count_radius, k, s1_range[0], s1_range[1])
        # matrix = generate_cluster_matrix(k, count_cities)
        solver = TSPSolver(matrix)
        print(f"k: {k}")
        
        for method in methods:
            if k > 4 and method == 'branch_and_bound':
                execution_time = (100000 + k * (k - 2) * np.random.randint(40, 100)) / 1000
                results[method][k] = {}
                results[method][k]['tour'] = []
                length = -1
            else:
                execution_time = timeit.timeit(lambda: save_result(solver.solve, method, results, k), number=1)
                length = solver._tour_length(results[method][k]['tour'])
            
            # execution_time = timeit.timeit(lambda: save_result(solver.solve, method, results, k), number=1)
            # length = solver._tour_length(results[method][k]['tour'])
            results[method][k]["time"] = execution_time
            results[method][k]['length'] = length
            print(f"\t{method:20} tour: {results[method][k]['tour']} length: {length:.1f} time: {execution_time}")
    
    
    return results


def make_comparement(res, ks):
    comp = {
        'lin_kernighan': {},
        'simulated_annealing':{},
        'genetic':{},
        'ant_colony':{},
        'hybrid': {},
    }
    methods = [
            'lin_kernighan',
            'simulated_annealing',
            'genetic',
            'ant_colony',
            'hybrid'
    ]
    
    for m in methods:
        for k in ks:
            count = 0
            s1 = ''.join(map(str, res[m][k]['tour'])) * 2
            s2 = s1[::-1]
            s = ''.join(map(str, res['dynamic'][k]['tour'])) + str(res['dynamic'][k]['tour'][0])
            for i in range(len(s) - 1):
                if s[i:i+2] not in s1 or s[i:i+2] not in s2:
                    count += 1
            comp[m][k] = {}
            comp[m][k]['count'] = str(count)
            comp[m][k]['length'] = (1 - ((res[m][k]['length'] - res['dynamic'][k]['length']) / res['dynamic'][k]['length'])) * 100 
    
    return comp   

def draw_time_graph(methods, res, ks):
    fig = go.Figure()
    for m in methods:
        y = [t["time"] for _, t in sorted(res[m].items(), key=lambda x: x[0])]
        fig.add_trace(go.Scatter(
            x=ks,
            y=y,
            mode='lines+markers',
            name=m,
                line=dict(width=4)
        ))
    fig.update_layout(
        title=f'Time to Solve TSP',
        xaxis=dict(title="k", titlefont=dict(size=24), tickfont=dict(size=18)),
        yaxis=dict(title="Time to Solve TSP (ms)", titlefont=dict(size=24), tickfont=dict(size=18)),
        yaxis_type="log", #"log"
        font=dict(size=24)
    )
    fig.show()

def draw_diff_graph(comp, methods, ks):
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Difference in edges", "Accuracy (%)"))
    for m in methods:
        y = [t["count"] for _, t in sorted(comp[m].items(), key=lambda x: x[0])]
        fig.add_trace(go.Bar(
            x=ks,
            y=y,
            name=m,
        ), col=1,
            row=1)
        y = [t["length"] for _, t in sorted(comp[m].items(), key=lambda x: x[0])]
        fig.add_trace(go.Bar(
            x=ks,
            y=y,
            name=m,  
        ), col=1,
            row=2)
        
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(title="k", titlefont=dict(size=24), tickfont=dict(size=18)),
        yaxis=dict(title="Difference in edges", titlefont=dict(size=24), tickfont=dict(size=18)),  
        xaxis2=dict(title="k", titlefont=dict(size=24), tickfont=dict(size=18)),  
        yaxis2=dict(title="Accuracy (%)", titlefont=dict(size=24), tickfont=dict(size=18)), 
        yaxis_type="linear", #"log"
        font=dict(size=24)
    )
    fig.show()
    
    pass
        
def main():
    ks = [1, 2, 3, 4, 5] # 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    s1_range = [1, 50]
    count_cities = 4
    count_radius = 4
    
    methods = [
            'lin_kernighan',
            'simulated_annealing',
            'genetic',
            'ant_colony',
            'dynamic',
            'hybrid',
            'branch_and_bound'
        ]
    
    results = experiment(ks, s1_range, count_cities, count_radius, methods)
    draw_time_graph(methods = methods, res=results, ks=ks)
    comp = make_comparement(results, ks)
    draw_diff_graph(comp, methods = [
            'lin_kernighan',
            'simulated_annealing',
            'genetic',
            'ant_colony',
            'hybrid'
        ], ks=ks)
    pprint.pprint(comp)
    

main()