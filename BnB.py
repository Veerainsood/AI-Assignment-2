import gymnasium as gym
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

# Global variables for DFBNB and animation frames
U = float('inf')   # Best (lowest) cost found so far (upper bound)
best_path = None   # Best path (list of actions) found so far
frames = []        # List to store frames for the GIF

def save_frames_as_gif(frames, path='./', filename='dfbnb_exploration.gif'):
    """Save a list of frames (numpy arrays) as a gif using ImageMagick."""
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
    
    # Increase the interval and lower fps for a slower animation.
    print("Saving the Animation....")
    print(len(frames))
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=80)
    anim.save(path + filename, writer='imagemagick', fps=30)
    print("Saved....")
    plt.close()

def reward_aware_heuristic(env, state, goal_state):
    """Heuristic function: Manhattan distance plus penalties/bonuses.
    
    Returns a low value for promising states, high value for holes, and negative for the goal.
    """
    grid_size = env.unwrapped.desc.shape[0]
    x1, y1 = divmod(state, grid_size)
    x2, y2 = divmod(goal_state, grid_size)
    
    # Basic Manhattan distance
    distance = abs(x1 - x2) + abs(y1 - y2)
    
    # Adjust heuristic: bonus for goal, heavy penalty for holes
    desc = env.unwrapped.desc.tolist()
    if desc[x1][y1] == b'G':  # Goal
        return -10
    elif desc[x1][y1] == b'H':  # Hole
        return 100
    return distance

def is_goal(state, env):
    """Check if the current state is the goal state."""
    grid_size = env.unwrapped.desc.shape[0]
    goal_state = grid_size * grid_size - 1
    return state == goal_state

def DFBNB(state, path, d, env):
    """
    Depth-First Branch and Bound recursive function.
    
    Parameters:
      - state: current state (integer for FrozenLake)
      - path: list of actions taken so far
      - d: dictionary mapping state -> best cost found so far to that state
      - env: the gym environment (with render_mode="rgb_array")
    """
    global U, best_path, frames

    # Capture a frame of the current state.
    frames.append(env.render())
    
    # If current state is goal, capture one final frame and update best solution.
    if is_goal(state, env):
        if d[state] <= U:
            U = d[state]
            best_path = path.copy()
            # Set environment state to goal and capture final frame.
            env.unwrapped.s = state
            frames.append(env.render())
            print(f"[INFO] Goal reached with cost {U}: {best_path}")
        return
    
    # Expand the current state.
    for action in range(env.action_space.n):
        # Save current state.
        saved_state = state
        env.unwrapped.s = saved_state
        
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Restore state after simulation.
        env.unwrapped.s = saved_state
        
        # The cost to move from state to next_state is 1.
        new_cost = d[state] + 1
        
        grid_size = env.unwrapped.desc.shape[0]
        goal_state_val = grid_size * grid_size - 1
        h_value = reward_aware_heuristic(env, next_state, goal_state_val)
        
        # Check bounding condition:
        if new_cost + h_value <= U and new_cost <= d.get(next_state, float('inf')):
            d[next_state] = new_cost
            DFBNB(next_state, path + [action], d, env)

def main(save=0):
    global U, best_path, frames
    # Create the FrozenLake environment in "rgb_array" mode for recording frames.
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array")
    
    # Reset the environment and initialize cost dictionary.
    initial_state, _ = env.reset()
    d = {initial_state: 0}
    
    start_time = time.time()
    DFBNB(initial_state, [], d, env)
    end_time = time.time()
    
    print(f"[INFO] Best path found: {best_path} with cost {U}")
    print(f"[INFO] Time taken: {end_time - start_time:.2f} seconds")
    
    # Save the collected frames as a GIF.
    if(save):
        imageio.mimsave(f'FrozenLake-DFBNB.gif', frames, duration=5)

    print("[INFO] Animation saved as dfbnb_exploration.gif")
    
    env.close()
    return end_time - start_time

def timed_run(i):
    return main(i)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    times = [timed_run(0) for i in range(5)]
    print("Run times:", times)
    print("Average time:", sum(times) / len(times))
    avg_times = []
    for i in range(1, 6):  # Iterations 1 to 5
        avg_times.append(sum(times[:i]) / len(times[:i]))

    print("Run times:", times)
    print("Average times:", avg_times)

    # Plotting
    plt.plot(range(1, 6), avg_times, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Average Time (s)')
    plt.title('Average Runtime vs Iterations (DFBNB on FrozenLake)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_time_vs_iterations.png")
    print("Plot saved as avg_time_vs_iterations.png")

        
