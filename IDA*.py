import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time
import imageio
import argparse
import matplotlib.pyplot as plt


def expand_actions(state):
    global dim
    actions = []
    x,y = state // dim, state % dim
    if y > 0:
        actions.append(0)
    if x < dim-1:
        actions.append(1)
    if y < dim-1:
        actions.append(2)
    if x > 0:
        actions.append(3)
    return actions

def IDA_Star_Driver(start_state, goal_state, heuristic, actions):
    global cost, bound, path_history,env,frames
    path_history = []
    r = False
    bound = heuristic(start_state, goal_state)
    frames = []
    while not r:
       cost = bound
       path_history = []
       bound = float('inf')
       r, path = IDA_Star(start_state, goal_state, 0, heuristic)
       frames.append(env.render())
       env.reset()
       #print("new bound: ", bound)
       
    return path

def IDA_Star(state, goal_state, current_cost, heuristic):
    global cost, bound, path_history,env,dim
    
    frames.append(env.render())
    if state == goal_state:
        return True, [state]

    allowed_actions = expand_actions(state)
    
    for action in allowed_actions:
        # print(env.unwrapped.s,"state ",current_cost,allowed_actions)
        new_state, reward, terminated, truncated, _ = env.step(action)
        # print(env.unwrapped.s,"state ",current_cost,new_state,action)
        # print("path_history",path_history)
        if terminated and new_state != goal_state:
            # print("Terminated",new_state,state)
            env.unwrapped.s = state
            # print(env.unwrapped.s)
            continue  

        if current_cost + 1 + heuristic(new_state, goal_state) <= cost:
            # print("Exploring new state:", new_state)
            path_history.append((state,action))
            # print("recu")
            r, path = IDA_Star(new_state, goal_state, current_cost + 1, heuristic)
            if r:
                return True, [state] + path
            path_history.pop()
        else:
            bound = min(bound, current_cost + 1 + heuristic(new_state, goal_state))
        env.unwrapped.s = state
    return False, []



def heuristic(state, goal_state):
    global dim
    return abs(state % dim - goal_state%dim) + abs(state // dim - goal_state// dim)

def get_actions():
    return [0, 1, 2, 3]

def get_start_state():
    return 0

def get_goal_state():
    global dim
    return dim*dim - 1

def main(n):
    global dim,path_history,frames
    frames = []
    dim = n
    start_state = get_start_state()
    goal_state = get_goal_state()
    actions = get_actions()
    print('goal_state: ',goal_state)
    times = []
    for i in range(5):
        start_time = time.time()
        path = IDA_Star_Driver(start_state, goal_state, heuristic, actions)
        end_time = time.time()
        times.append(end_time - start_time)
    average_time = sum(times)/len(times)
    print("Average time taken: ", average_time)
    print(path)
    print(len(path))
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, 6), times, marker='o')
    # plt.axhline(y=average_time, color='r', linestyle='--', label='Average Time')
    # plt.legend()
    # plt.title("Time vs Runs")
    # plt.xlabel("Run")
    # plt.ylabel("Time (seconds)")
    # plt.xticks(range(1, 6))
    # plt.grid()
    # plt.savefig(f'FrozenLake-IDA*{dim}x{dim}'+"_time_vs_runs.png")
    print("path history",path_history)
    print("Time taken: ", end_time - start_time)
    imageio.mimsave(f'FrozenLake-IDA*{dim}x{dim}.gif', frames, duration=5)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=8)
    parser.add_argument("--r", type=bool,default=False)
    args = parser.parse_args()
    isRandom = args.r
    dim = args.s
    env = gym.make('FrozenLake-v1',is_slippery=False, desc=generate_random_map(size=dim), render_mode="rgb_array") if isRandom else gym.make('FrozenLake-v1', is_slippery=False, map_name=f'{dim}x{dim}', render_mode="rgb_array")
    env.reset()
    main(dim)