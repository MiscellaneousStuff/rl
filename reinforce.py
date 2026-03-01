import gymnasium as gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

OPTIM = "ADAM" # "ADAM"

class REINFORCE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

if __name__ == "__main__":
    env_name = "CartPole-v1"

    env = gym.make(env_name, render_mode="None")
    obs, info = env.reset(seed=SEED)

    total_reward = 0
    ep = 0
    ep_rewards = [0]
    start_eps = 1.0 # chance of exploit (explore = 1 / eps)
    steps = 100000 * 1

    model = REINFORCE(in_dim = 4, out_dim = 2)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    gamma = 0.99
    warmup_steps = 512
    update_steps = 4
    update_values = []
    update_targets = []
    max_buffer = 1024 * 10
    target_steps = 512

    acts = []
    rewards = []
    obs_s = []

    for step in range(steps):
        eps = (start_eps - (step / steps))
        # eps = 1.0
        # action = env.action_space.sample()  # replace with your policy
        # bin_obs = convert_obs(obs, N=bins)
        obs = torch.tensor(obs)
        act_probs = model(obs)

        dist = torch.distributions.Categorical(act_probs)
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        action = action_tensor.item()  # for env.step

        new_obs, reward, terminated, truncated, info = env.step(action)

        obs_s.append(obs)
        acts.append(log_prob)
        rewards.append(reward)

        # update(bin_obs, table, action, reward, convert_obs(obs, N=bins), terminated)
        # update NN here
        # new_obs = torch.tensor(new_obs)
        # future_values = model(new_obs)
        # max_val = torch.max(future_values.detach().clone())

        # target = reward + gamma * max_val
        
        # if len(update_values) == max_buffer:
        #     update_values.pop(0)
        #     update_targets.pop(0)
        # update_values.append([obs, action, reward, new_obs, terminated]) # values[action])
        # update_targets.append(target)

        # if step % target_steps == 0:
        #     target_model.load_state_dict(model.state_dict())

        # if step % update_steps == 0 and step != 0 and step > warmup_steps:
        #     # print("len(update_values):", len(update_values))
        #     # print(update_values)
        #     # perm = torch.randperm(len(update_values))
        #     # update_values_stack = torch.stack(update_values)[perm][0:32]
        #     # update_targets_stack = torch.stack(update_targets)[perm][0:32]
        #     indices = random.sample(range(len(update_values)), 32)
        #     update_values_batch = [update_values[i] for i in indices]
        #     update_targets_batch = [update_targets[i] for i in indices]

        #     obs_batch = torch.stack([b[0] for b in update_values_batch])
        #     act_batch = [b[1] for b in update_values_batch]
        #     rew_batch = torch.tensor([b[2] for b in update_values_batch], dtype=torch.float32)
        #     next_obs_batch = torch.stack([b[3] for b in update_values_batch])
        #     done_batch = torch.tensor([b[4] for b in update_values_batch], dtype=torch.float32)
            
        #     q_values = model(obs_batch)  # [32, 2]
        #     q_sa = q_values[range(32), act_batch]  # Q(s, a) for the taken action

        #     with torch.no_grad():
        #         next_q = target_model(next_obs_batch)
        #         max_next_q = next_q.max(dim=1).values
        #         targets = rew_batch + gamma * max_next_q * (1 - done_batch)

        #     # print("update_values.shape, update_targets.shape:", update_values.shape, update_targets.shape)

        #     optim.zero_grad()
        #     if OPTIM == "MUON":
        #         optim_bias.zero_grad()
        #     loss = F.mse_loss(q_sa, targets)
        #     loss.backward()
        #     optim.step()
        #     if OPTIM == "MUON":
        #         optim_bias.step()
            
        #     # update_values = []
        #     # update_targets = []

        total_reward += reward

        obs = new_obs 

        if terminated or truncated:
            obs, info = env.reset()

            act_log_probs = torch.stack(acts)
            discounted_rewards = []
            for i in range(0, len(rewards)):
                G = 0
                for j in range(i, len(rewards)):
                    G += rewards[j] * (gamma ** (j - i))
                discounted_rewards.append(G)

            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
            # print(act_log_probs.shape, discounted_rewards.shape)
            loss = (-act_log_probs * discounted_rewards).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()

            obs_s = []
            acts = []
            rewards = []

            if ep % 1000 == 0:
                print(ep, total_reward, eps, np.max(ep_rewards), np.mean(ep_rewards))
            ep += 1
            
            ep_rewards.append(total_reward)
            # if (total_reward / np.mean(ep_rewards)) > 6:
            #     break
            total_reward = 0

    env.close()

    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset(seed=SEED)

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