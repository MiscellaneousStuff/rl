import gymnasium as gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import random

def update(bin_obs, table, act, rew, next_obs, terminated, lr=0.1, gamma=0.99):
    """NOTE: obs := `bin_obs`"""
    # 1. select (s, a) row
    obs_mask = np.all(table[:, 0:4] == bin_obs, axis=1)
    q_sa = table[obs_mask, 4+act]

    # 2. select max action over (s')
    next_obs_mask = np.all(table[:, 0:4] == next_obs, axis=1)
    best_val = np.max(table[next_obs_mask, 4:6], axis=-1)
    # print("best_val", table[next_obs_mask, 4:6].shape, best_val.shape, best_val)

    # 2. (s, a) = r
    try:
        if not terminated:
            table[obs_mask, 4+act] = q_sa + lr * (rew + gamma*best_val - q_sa)
        else:
            table[obs_mask, 4+act] = q_sa + lr * (rew - q_sa)
    except ValueError as e:
        print(e)

    # print("post", bin_obs, table[mask], act, rew)

def act(table, obs, N=8):
    bin_obs = convert_obs(obs, N=N)
    # print("bin_obs:", bin_obs)
    mask = np.all(table[:, 0:4] == bin_obs, axis=1)
    # print(bin_obs.shape, table[:, 0:4].shape)
    act_vals = table[mask]
    # print(act_vals.shape)
    act_vals = act_vals[0, 4:6]
    act_idx = np.argmax(act_vals)
    return act_idx

def q_table(N=3):
    # cart_pos cart_vel pole_ang pole_ang_vel left right
    obs_vals = list(range(N))
    obs_arr  = np.array(list(product(obs_vals, repeat=4)))
    act_arr  = np.zeros((obs_arr.shape[0], 2))
    q_table = np.concatenate((obs_arr, act_arr), axis=1)
    return q_table

def convert_obs(obs, N=5):
    cart_pos, cart_vel, pole_ang, pole_ang_vel = obs
    def bin(x, max, N=5):
        bins = (x + max) / (max * 2)
        return np.minimum((bins * N).astype(int), N-1) # [0, N-1]
    
    cart_pos     = bin(cart_pos, 4.8, N)
    cart_vel     = bin(cart_vel, 6.0, N)
    pole_ang     = bin(pole_ang, 0.418, N)
    pole_ang_vel = bin(pole_ang_vel, 6.0, N)

    return np.array([[cart_pos, cart_vel, pole_ang, pole_ang_vel]])

if __name__ == "__main__":
    bins = 8
    table = q_table(N=bins)

    env = gym.make("CartPole-v1", render_mode="None")
    obs, info = env.reset(seed=42)

    total_reward = 0
    ep = 0
    ep_rewards = [0]
    start_eps = 1.0 # chance of exploit (explore = 1 / eps)
    steps = 100_000

    for step in range(steps):
        eps = (start_eps - (step / steps))
        # eps = 1.0
        # action = env.action_space.sample()  # replace with your policy
        bin_obs = convert_obs(obs, N=bins)
        if random.random() > eps:
            action = act(table, obs, N=bins)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        update(bin_obs, table, action, reward, convert_obs(obs, N=bins), terminated)
        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()
            if ep % 1000 == 0:
                print(ep, total_reward, eps, np.max(ep_rewards), np.mean(ep_rewards))
            ep += 1
            
            ep_rewards.append(total_reward)
            total_reward = 0

    env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset(seed=42)

    total_reward = 0
    for steps in range(500):
        bin_obs = convert_obs(obs, N=bins)
        if random.random() > eps:
            action = act(table, obs, N=bins)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()
            print(ep, total_reward)
            total_reward = 0

    plt.plot(ep_rewards)
    plt.show()
    env.close()