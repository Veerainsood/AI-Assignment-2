import gymnasium as gym
import numpy as np
import time
import imageio
import copy

frames = []

def expand_actions(state):
    actions = []
    x,y = state // 4, state % 4
    if y > 0:
        actions.append(0)
    if x < 3:
        actions.append(1)
    if y < 3:
        actions.append(2)
    if x > 0:
        actions.append(3)
    return actions

def IDA_Star_Driver(start_state, goal_state, heuristic, actions):
    global cost, bound, path_history,env
    path_history = []
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name='4x4', render_mode="rgb_array")
    env.reset()
    r = False
    bound = heuristic(start_state, goal_state)
    while not r:
       cost = bound
       bound = float('inf')
       r, path = IDA_Star(start_state, goal_state, 0, heuristic)
       env.reset()
       print("new bound: ", bound)
    return path

def IDA_Star(state, goal_state, current_cost, heuristic):
    global cost, bound, path_history,env
    frames.append(env.render())
    
    if state == goal_state:
        return True, [state]

    allowed_actions = expand_actions(state)
    
    for action in allowed_actions:
        # print(env.unwrapped.s,"state ",current_cost,allowed_actions)
        new_state, reward, terminated, truncated, _ = env.step(action)
        # print(env.unwrapped.s,"state ",current_cost,new_state,action)
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
    return abs(state % 4 - goal_state%4) + abs(state // 4 - goal_state// 4)

def get_actions():
    return [0, 1, 2, 3]

def get_start_state():
    return 0

def get_goal_state():
    return 15

def main():
    start_state = get_start_state()
    goal_state = get_goal_state()
    actions = get_actions()
    start_time = time.time()
    print('goal_state: ',goal_state)
    path = IDA_Star_Driver(start_state, goal_state, heuristic, actions)
    end_time = time.time()
    print(path)
    print("Time taken: ", end_time - start_time)
    imageio.mimsave('FrozenLake-IDA*.gif', frames, duration=0.5)
    
main()