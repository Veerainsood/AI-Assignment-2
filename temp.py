import numpy as np
import gym

def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour) - 1)) + dist_matrix[tour[-1], tour[0]]

def generate_neighbors(tour, num_neighbors=5):
    neighbors = []
    for _ in range(num_neighbors):
        new_tour = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        neighbors.append(new_tour)
    return neighbors

def simulated_annealing_tsp(env, T=30, alpha=0.995, min_T=1e-3, num_candidates=5, max_iters=100000):
    state = env.reset()
    dist_matrix = env.get_attr("distances")[0]  # Extract distance matrix from environment
    n_cities = len(dist_matrix)
    
    current_tour = list(range(n_cities))
    np.random.shuffle(current_tour)
    current_cost = total_distance(current_tour, dist_matrix)

    best_tour, best_cost = current_tour, current_cost

    while T > min_T and max_iters > 0:
        candidates = generate_neighbors(current_tour, num_candidates)
        candidate_costs = [total_distance(tour, dist_matrix) for tour in candidates]
        
        best_candidate_idx = np.argmin(candidate_costs)
        best_candidate, best_candidate_cost = candidates[best_candidate_idx], candidate_costs[best_candidate_idx]
        
        delta = best_candidate_cost - current_cost
        if delta < 0 or np.exp(-delta / T) > np.random.rand():
            current_tour, current_cost = best_candidate, best_candidate_cost

            if current_cost < best_cost:
                best_tour, best_cost = current_tour, current_cost

        T *= alpha
        max_iters -= 1

    return best_tour, best_cost

# Running in the gym_TSP environment
env = gym.make("tsp-v0")
best_tour, best_cost = simulated_annealing_tsp(env)
print("Optimized Tour:", best_tour)
print("Total Distance:", best_cost)
