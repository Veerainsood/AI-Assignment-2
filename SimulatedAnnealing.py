import math
import random
import time
import os
import networkx as nx
import matplotlib.pyplot as plt
import imageio

# === Step 1: Parse .tsp file ===
def parse_tsp_file(file_path):
    coords = []
    best_known = None
    with open(file_path, 'r') as f:
        lines = f.readlines()
        in_coords = False
        for line in lines:
            line = line.strip()
            if line.startswith("BEST_KNOWN"):
                best_known = int(line.split(":")[1].strip())
            if line.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            if in_coords:
                if line == "EOF" or not line:
                    break
                parts = line.split()
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))
    return coords, best_known

# === Step 2: Distance + cost ===
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def total_distance(tour, coords):
    return sum(euclidean(coords[tour[i - 1]], coords[tour[i]]) for i in range(len(tour)))

# === Step 3: Christofides Initial Tour ===
def christofides_tour(coords):
    n = len(coords)
    G = nx.Graph()

    # Build complete graph with Euclidean distances as weights
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean(coords[i], coords[j])
            G.add_edge(i, j, weight=dist)

    # 1. Minimum Spanning Tree
    T = nx.minimum_spanning_tree(G)

    # 2. Find vertices with odd degree
    odd_degree_nodes = [v for v in T.nodes if T.degree[v] % 2 == 1]

    # 3. Induce subgraph of odd-degree nodes with correct edge weights
    subgraph = nx.Graph()
    for u in odd_degree_nodes:
        for v in odd_degree_nodes:
            if u < v:
                dist = euclidean(coords[u], coords[v])
                subgraph.add_edge(u, v, weight=dist)

    # 4. Minimum weight perfect matching
    matching = nx.algorithms.matching.min_weight_matching(subgraph)

    # 5. Multigraph union of MST + matching
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(T.edges(data=True))
    multigraph.add_edges_from(((u, v, {'weight': euclidean(coords[u], coords[v])}) for u, v in matching))

    # 6. Find Eulerian circuit
    euler_circuit = list(nx.eulerian_circuit(multigraph))

    # 7. Make Hamiltonian tour by shortcutting repeated nodes
    visited = set()
    tour = []
    for u, v in euler_circuit:
        if u not in visited:
            tour.append(u)
            visited.add(u)
        if v not in visited:
            tour.append(v)
            visited.add(v)

    return tour


# === Step 4: 3-opt ===
def three_opt(tour, coords):
    n = len(tour)
    a, b, c = sorted(random.sample(range(n), 3))

    def dist(i, j):
        return euclidean(coords[tour[i % n]], coords[tour[j % n]])

    segments = [
        tour[:a+1] + tour[b+1:c+1] + tour[a+1:b+1] + tour[c+1:],
        tour[:a+1] + tour[b+1:c+1] + tour[b:a:-1] + tour[c+1:],
        tour[:a+1] + tour[c:b:-1] + tour[a+1:b+1] + tour[c+1:],
        tour[:a+1] + tour[c:b:-1] + tour[b:a:-1] + tour[c+1:],
        tour[:a+1] + tour[b+1:c+1][::-1] + tour[a+1:b+1] + tour[c+1:],
        tour[:a+1] + tour[b+1:c+1][::-1] + tour[b:a:-1] + tour[c+1:],
        tour[:a+1] + tour[b+1:c+1][::-1] + tour[a+1:b+1][::-1] + tour[c+1:]
    ]

    best = tour
    best_cost = total_distance(tour, coords)
    for candidate in segments:
        cand_cost = total_distance(candidate, coords)
        if cand_cost < best_cost:
            best = candidate
            best_cost = cand_cost
    return best

# === Step 5: Simulated Annealing + GIF ===
def simulated_annealing_with_gif(coords, initial_tour, max_time_sec=180, save_gif="tour_evolution.gif"):
    current_tour = initial_tour
    current_cost = total_distance(current_tour, coords)
    best_tour = list(current_tour)
    best_cost = current_cost
    temperature = 10000.0
    alpha = 0.9995
    stop_temp = 1e-4
    max_iter = 100000

    tour_snapshots = []
    times = []
    start_time = time.time()

    for iteration in range(max_iter):
        now = time.time()
        if temperature < stop_temp or (now - start_time) > max_time_sec:
            print(f"Stopped at iteration {iteration}, time used: {now - start_time:.2f}s")
            break

        candidate = three_opt(current_tour, coords)
        candidate_cost = total_distance(candidate, coords)
        delta = candidate_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_tour = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour

        temperature *= alpha

        if iteration % 1000 == 0:
            print(f"[Iter {iteration}] Temp: {temperature:.2f}, Best: {best_cost:.2f}")
            tour_snapshots.append(list(best_tour))
            times.append(now - start_time)

    make_gif(coords, tour_snapshots, save_gif)
    return best_tour, best_cost

# === Step 6: Make GIF ===
def make_gif(coords, tours, filename):
    images = []
    for tour in tours:
        fig, ax = plt.subplots()
        x = [coords[city][0] for city in tour]
        y = [coords[city][1] for city in tour]
        ax.plot(x, y, 'bo-', markersize=2, linewidth=1)
        ax.set_title("TSP Tour Optimization")
        ax.axis('off')
        fig.tight_layout()
        img_path = "frame.png"
        plt.savefig(img_path)
        plt.close()
        images.append(imageio.imread(img_path))
        os.remove(img_path)
    imageio.mimsave(filename, images, fps=5)
    print(f"Saved GIF to {filename}")

# === Step 7: Run Everything ===
best_known = 0
def solve_tsp_file(path):
    global best_known

    print(f"Loading: {path}")
    coords, best_known = parse_tsp_file(path)
    print(f"Dimension: {len(coords)} | Best Known: {best_known}")

    initial_tour = christofides_tour(coords)
    initial_tour.append(initial_tour[0]) 
    print("Initial Christofides tour length:", total_distance(initial_tour, coords))

    best_tour, best_cost = simulated_annealing_with_gif(coords, initial_tour)

    print("\n=== Final Result ===")
    print(f"Best tour distance found: {best_cost:.2f}")
    if best_known:
        print(f"Gap from best known: {((best_cost - best_known) / best_known) * 100:.2f}%")
    return best_tour, best_cost

# === Main ===
if __name__ == "__main__":
    tsp_paths = ["problems_cleaned/fl1577.tsp" ,
                "problems_cleaned/eil76.tsp" ,
                "problems_cleaned/ch130.tsp" ,
                "problems_cleaned/pcb442.tsp" ,
                "problems_cleaned/rat783.tsp" 
                ]
    
    avg_Gap = 0

    for i in range(len(tsp_paths)):
        _ , best_cost = solve_tsp_file(tsp_paths[i])
        avg_Gap += ((best_cost - best_known) / best_known) * 100
    
    print(f"Average Gap from best known: {avg_Gap/len(tsp_paths):.2f}%")
    