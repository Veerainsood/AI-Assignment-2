import math
import random
import time
import os
import networkx as nx
import matplotlib.pyplot as plt
import imageio.v2 as imageio

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
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean(coords[i], coords[j])
            G.add_edge(i, j, weight=dist)
    T = nx.minimum_spanning_tree(G)
    odd_nodes = [v for v in T.nodes if T.degree[v] % 2 == 1]
    subgraph = nx.Graph()
    for u in odd_nodes:
        for v in odd_nodes:
            if u < v:
                subgraph.add_edge(u, v, weight=euclidean(coords[u], coords[v]))
    matching = nx.algorithms.matching.min_weight_matching(subgraph)
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(T.edges(data=True))
    multigraph.add_edges_from(((u, v, {'weight': euclidean(coords[u], coords[v])}) for u, v in matching))
    euler_circuit = list(nx.eulerian_circuit(multigraph))
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

# === Step 5: Simulated Annealing with Stats ===
def simulated_annealing_with_gif(coords, initial_tour, max_time_sec=180, save_gif="tour_evolution.gif",plot=True):
    current_tour = initial_tour
    current_cost = total_distance(current_tour, coords)
    best_tour = list(current_tour)
    best_cost = current_cost
    temperature = 10000.0
    alpha = 0.9995
    stop_temp = 1e-4
    max_iter = 100000

    tour_snapshots = []
    cost_list = []
    temperature_list = []
    acceptance_prob_list = []

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
                best_tour = list(current_tour)
            acceptance_prob = 1 if delta < 0 else math.exp(-delta / temperature)
        else:
            acceptance_prob = 0

        # Stats logging
        cost_list.append(current_cost)
        temperature_list.append(temperature)
        acceptance_prob_list.append(acceptance_prob)

        temperature *= alpha

        if (iteration % 1000 == 0 or current_cost < best_cost) and plot:
            print(f"[Iter {iteration}] Temp: {temperature:.2f}, Best: {best_cost:.2f}")
            tour_snapshots.append(list(best_tour))
            # Also log up to this step so plots align
    if plot:
        make_combined_gif(coords, tour_snapshots, cost_list, temperature_list, acceptance_prob_list, save_gif)
    
    # ✅ Return all values needed
    return best_tour, best_cost, cost_list, temperature_list, acceptance_prob_list


# === Step 6: Make GIF ===
def make_combined_gif(coords, tours, costs, temps, probs, filename):
    import matplotlib.pyplot as plt
    import imageio
    import os

    images = []
    for i in range(len(tours)):
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))

        tour = tours[i]
        x = [coords[city][0] for city in tour]
        y = [coords[city][1] for city in tour]
        step = (i + 1) * 1000

        # Top-left: Tour Plot
        axs[0, 0].plot(x, y, 'bo-', markersize=2, linewidth=1)
        axs[0, 0].set_title(f"TSP Tour (Step {i * 1000})")
        axs[0, 0].axis('off')

        # Top-right: Cost
        axs[0, 1].plot(costs[:step], color='blue')
        axs[0, 1].set_title("Total Length (Cost)")
        axs[0, 1].set_xlabel("Step")
        axs[0, 1].set_ylabel("Distance")

        # Bottom-left: Temperature
        axs[1, 0].plot(temps[:step], color='orange')
        axs[1, 0].set_title("Temperature")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Temp")

        # Bottom-right: Acceptance Probability
        axs[1, 1].scatter(range(len(probs[:step])), probs[:step], s=2, alpha=0.4, color='green')
        axs[1, 1].set_title("Acceptance Probability")
        axs[1, 1].set_xlabel("Step")
        axs[1, 1].set_ylabel("Prob")

        fig.tight_layout()
        temp_file = "frame_tmp.png"
        plt.savefig(temp_file)
        plt.close()
        images.append(imageio.imread(temp_file))
        os.remove(temp_file)

    imageio.mimsave(filename, images, fps=5)
    print(f"✅ Saved combined GIF: {filename}")



# === Step 7: Summary Plot ===
def save_summary_plot(costs, temps, probs, filename):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
    axs[0, 0].plot(costs)
    axs[0, 0].set_title("Total length (cost)")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("Length")

    axs[0, 1].plot(temps, color='orange')
    axs[0, 1].set_title("Temperature")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Temperature")

    axs[1, 0].scatter(range(len(probs)), probs, s=2)
    axs[1, 0].set_title("Acceptance probability")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("Probability")

    axs[1, 1].axis('off')  # Optional: add more plots/info

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved summary plot to {filename}")

# === Step 8: Run Everything ===
best_known = 0
def solve_tsp_file(path,plot=True):
    global best_known

    print(f"Loading: {path}")
    coords, best_known = parse_tsp_file(path)
    print(f"Dimension: {len(coords)} | Best Known: {best_known}")

    initial_tour = christofides_tour(coords)
    initial_tour.append(initial_tour[0])
    print("Initial Christofides tour length:", total_distance(initial_tour, coords))

    base_name = os.path.basename(path).replace(".tsp", "")
    gif_name = f"{base_name}_evolution.gif"
    summary_plot_name = f"{base_name}_summary.png"

    best_tour, best_cost, costs, temps, probs = simulated_annealing_with_gif(
        coords, initial_tour, save_gif=gif_name,plot=plot
    )
    if plot:
        save_summary_plot(costs, temps, probs, summary_plot_name)

    print("\n=== Final Result ===")
    print(f"Best tour distance found: {best_cost:.2f}")
    if best_known:
        print(f"Gap from best known: {((best_cost - best_known) / best_known) * 100:.2f}%")
    return best_tour, best_cost

# === Main ===
if __name__ == "__main__":
    tsp_paths = [
        "problems_cleaned/eil76.tsp",
        "problems_cleaned/ch130.tsp",
        "problems_cleaned/d198.tsp",
        "problems_cleaned/kroA100.tsp",
    ]
    
    avg_Gap = 0
    for path in tsp_paths:
        if(path == "problems_cleaned/eil76.tsp"):
            times = []
            avg_times = []
            for i in range(5):
                start = time.time()
                _, best_cost = solve_tsp_file(path,plot=False)
                stop = time.time()
                times.append(stop-start)
                avg_times.append(sum(times)/len(times))
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, 6), avg_times, marker='o', linestyle='-', color='blue')
            plt.title("Average Time vs Iteration (eil76.tsp)")
            plt.xlabel("Run Number")
            plt.ylabel("Average Time (seconds)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("S_A_avg_time_vs_iterations.png")
            solve_tsp_file(path)

        _, best_cost = solve_tsp_file(path)
        avg_Gap += ((best_cost - best_known) / best_known) * 100

    print(f"\nAverage Gap from best known: {avg_Gap / len(tsp_paths):.2f}%")
