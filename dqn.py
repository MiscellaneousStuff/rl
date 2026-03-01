import gymnasium as gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkLinear(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = 8):
        super().__init__()
        # self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        # x = F.relu(self.fc1(obs))
        # x = self.fc2(x)
        x = self.fc(x)
        return x

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
    # table = q_table(N=bins)

    env = gym.make("CartPole-v1", render_mode="None")
    obs, info = env.reset(seed=42)

    total_reward = 0
    ep = 0
    ep_rewards = [0]
    start_eps = 1.0 # chance of exploit (explore = 1 / eps)
    steps = 100000

    model = QNetworkLinear(in_dim = 4, out_dim = 2)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    gamma = 0.99
    warmup_steps = 512
    update_steps = 4
    update_values = []
    update_targets = []
    max_buffer = 1024 * 10

    for step in range(steps):
        eps = (start_eps - (step / steps))
        # eps = 1.0
        # action = env.action_space.sample()  # replace with your policy
        # bin_obs = convert_obs(obs, N=bins)
        obs = torch.tensor(obs)
        values = model(obs)

        if random.random() > eps:
            # action = act(table, obs, N=bins)
            action = torch.argmax(values).item()
            # print("ACT:", action)
        else:
            action = env.action_space.sample()

        new_obs, reward, terminated, truncated, info = env.step(action)

        # update(bin_obs, table, action, reward, convert_obs(obs, N=bins), terminated)
        # update NN here
        new_obs = torch.tensor(new_obs)
        future_values = model(new_obs)
        max_val = torch.max(future_values.detach().clone())

        target = reward + gamma * max_val
        
        if len(update_values) == max_buffer:
            update_values.pop(0)
            update_targets.pop(0)
        update_values.append([obs, action, reward, new_obs, terminated]) # values[action])
        update_targets.append(target)

        if step % update_steps == 0 and step != 0 and step > warmup_steps:
            # print("len(update_values):", len(update_values))
            # print(update_values)
            # perm = torch.randperm(len(update_values))
            # update_values_stack = torch.stack(update_values)[perm][0:32]
            # update_targets_stack = torch.stack(update_targets)[perm][0:32]
            indices = random.sample(range(len(update_values)), 32)
            update_values_batch = [update_values[i] for i in indices]
            update_targets_batch = [update_targets[i] for i in indices]

            obs_batch = torch.stack([b[0] for b in update_values_batch])
            act_batch = [b[1] for b in update_values_batch]
            rew_batch = torch.tensor([b[2] for b in update_values_batch], dtype=torch.float32)
            next_obs_batch = torch.stack([b[3] for b in update_values_batch])
            done_batch = torch.tensor([b[4] for b in update_values_batch], dtype=torch.float32)
            
            q_values = model(obs_batch)  # [32, 2]
            q_sa = q_values[range(32), act_batch]  # Q(s, a) for the taken action

            with torch.no_grad():
                next_q = model(next_obs_batch)
                max_next_q = next_q.max(dim=1).values
                targets = rew_batch + gamma * max_next_q * (1 - done_batch)

            # print("update_values.shape, update_targets.shape:", update_values.shape, update_targets.shape)
            optim.zero_grad()
            loss = F.mse_loss(q_sa, targets)
            loss.backward()
            optim.step()
            
            # update_values = []
            # update_targets = []

        total_reward += reward

        obs = new_obs 
        
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
        obs = torch.tensor(obs)
        values = model(obs)
        action = torch.argmax(values).item()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()
            print(ep, total_reward)
            total_reward = 0

    plt.plot(ep_rewards)
    plt.show()
    env.close()