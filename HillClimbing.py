import numpy as np
import time
import matplotlib.pyplot as plt
import imageio
import os

def parse_tsp_file(filepath):
    cities = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        node_section = False
        for line in lines:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                node_section = True
                continue
            if line == "EOF":
                break
            if node_section:
                parts = line.split()
                x, y = float(parts[1]), float(parts[2])
                cities.append([x, y])
    return np.array(cities)

def distance_matrix(cities):
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def greedy_initial_tour(dist_matrix):
    """
    Constructs an initial tour using a greedy nearest-neighbor heuristic.

    Args:
        dist_matrix (np.ndarray): Precomputed distance matrix.

    Returns:
        list: Greedy initial tour.
    """
    n = len(dist_matrix)
    unvisited = set(range(n))
    current_city = 0 
    tour = [current_city]
    unvisited.remove(current_city)

    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current_city][city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return tour

def total_distance(tour, dist_matrix):
    return np.sum(dist_matrix[tour, np.roll(tour, -1)])

def get_best_2opt_neighbor(tour, dist_matrix):
    best_cost = total_distance(tour, dist_matrix)
    best_tour = tour
    for i in range(len(tour) - 1):
        for j in range(i + 2, len(tour)):
            if j - i == 1:
                continue
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
            cost = total_distance(new_tour, dist_matrix)
            if cost < best_cost:
                best_cost = cost
                best_tour = new_tour
    return best_tour, best_cost

def hill_climb_with_restarts(cities, restarts=5, max_iterations=1000, seed=42):
    best_overall_tour = None
    best_overall_cost = float('inf')
    dist_matrix = distance_matrix(cities)
    frame_count = 0

    for r in range(restarts):
        print(f"Restart {r + 1}/{restarts}")
        np.random.seed(seed + r)
        current_tour = greedy_initial_tour(dist_matrix)
        current_cost = total_distance(current_tour, dist_matrix)

        for i in range(max_iterations):
            print("current cost:", current_cost)
            print("current tour:", current_tour)
            next_tour, next_cost = get_best_2opt_neighbor(current_tour, dist_matrix)
            if next_cost < current_cost:
                current_tour, current_cost = next_tour, next_cost

                # Plot only on improvement
                plt.figure(figsize=(8, 6))
                plt.scatter(cities[:, 0], cities[:, 1], color='blue')
                path = current_tour + [current_tour[0]]
                plt.plot(cities[path, 0], cities[path, 1], color='red')
                plt.title(f"Tour Improvement (Restart {r + 1}, Iter {i + 1})")
                plt.xlim(np.min(cities[:, 0]) - 10, np.max(cities[:, 0]) + 10)
                plt.ylim(np.min(cities[:, 1]) - 10, np.max(cities[:, 1]) + 10)
                filename = f"frame_{frame_count}.png"
                plt.savefig(filename)
                plt.close()
                frame_count += 1
            else:
                break

        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_tour = current_tour

    return best_overall_tour, best_overall_cost, frame_count

# --- Main Execution ---
cities = parse_tsp_file("TSP.txt")
plt.scatter(cities[:, 0], cities[:, 1])
plt.title("City Locations")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.xlim(np.min(cities[:, 0]) - 10, np.max(cities[:, 0]) + 10)
plt.ylim(np.min(cities[:, 1]) - 10, np.max(cities[:, 1]) + 10)
plt.savefig("cities.png")
plt.show()

start_time = time.time()
best_tour, best_cost, total_frames = hill_climb_with_restarts(cities)
end_time = time.time()

print("Best tour:", best_tour)
print("Best cost:", best_cost)
print("Time taken:", end_time - start_time)

# --- Create GIF ---
images = []
for i in range(total_frames):
    filename = f"frame_{i}.png"
    images.append(imageio.imread(filename))
imageio.mimsave('hill_climb.gif', images, fps=1)

print(f"GIF created with {total_frames} frames.")
