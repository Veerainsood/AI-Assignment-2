import numpy as np
import time
import matplotlib.pyplot as plt
import imageio
import argparse
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

def hill_climb_with_restarts(cities, restarts=1, max_iterations=1000, seed=42):
    best_overall_tour = None
    best_overall_cost = float('inf')
    dist_matrix = distance_matrix(cities)
    frame_count = 0

    for r in range(restarts):
        # print(f"Restart {r + 1}/{restarts}")
        np.random.seed(seed + r)
        current_tour = greedy_initial_tour(dist_matrix)
        current_cost = total_distance(current_tour, dist_matrix)

        for i in range(max_iterations):
            # print("current cost:", current_cost)
            # print("current tour:", current_tour)
            next_tour, next_cost = get_best_2opt_neighbor(current_tour, dist_matrix)
            if next_cost < current_cost:
                current_tour, current_cost = next_tour, next_cost
                # Plot only on improvement
                plt.figure(figsize=(8, 6))
                plt.scatter(cities[:, 0], cities[:, 1], color='blue')
                path = current_tour + [current_tour[0]]
                plt.plot(cities[path, 0], cities[path, 1], color='red')
                plt.title(f"Tour Improvement (Iter {i + 1})")
                plt.xlim(np.min(cities[:, 0]) - 10, np.max(cities[:, 0]) + 10)
                plt.ylim(np.min(cities[:, 1]) - 10, np.max(cities[:, 1]) + 10)
                filename = f"./ignore/frame_{frame_count}.png"
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
if __name__ == "__main__":
    # Ensure the output directory exists
    parser = argparse.ArgumentParser()
    parser.add_argument("--f" ,type=int, default=1)
    args = parser.parse_args()
    os.makedirs("./ignore", exist_ok=True)
    files = ["./problems_cleaned/ch130.tsp","./problems_cleaned/eil76.tsp","./problems_cleaned/d198.tsp/","./problems_cleaned/kroA100.tsp"]

    # Parse TSP file and plot cities
    cities = parse_tsp_file(files[args.f])
    plt.scatter(cities[:, 0], cities[:, 1])
    plt.title("City Locations")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(np.min(cities[:, 0]) - 10, np.max(cities[:, 0]) + 10)
    plt.ylim(np.min(cities[:, 1]) - 10, np.max(cities[:, 1]) + 10)
    plt.savefig("cities.png")
    plt.show()
    times = []
    for i in range(5):
        start_time = time.time()
        best_tour, best_cost, total_frames = hill_climb_with_restarts(cities)
        end_time = time.time()

        print("Best tour:", best_tour)
        print("Best cost:", best_cost)
        print("Time taken:", end_time - start_time)
        times.append(end_time - start_time)

    average_time = sum(times) / len(times)
    #plot time vs  runs
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 6), times, marker='o')
    plt.axhline(y=average_time, color='r', linestyle='--', label='Average Time')
    plt.legend()
    plt.title("Time vs Runs")
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.xticks(range(1, 6))
    plt.grid()
    plt.savefig(files[args.f].split("/")[-1].split(".")[0]+" time_---vs_runs.png")
    # --- Create GIF ---
    gif_name = f'hill_climb_{files[args.f].split("/")[-1].split(".")[0]}.gif'
    images = []
    for i in range(total_frames):
        filename = f"./ignore/frame_{i}.png"
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_name, images, fps=1)
    print(f"GIF created with {total_frames} frames.")



    
