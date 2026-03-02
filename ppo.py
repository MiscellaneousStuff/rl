import gymnasium as gym
import numpy as np
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

class ActorCritic(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Linear(in_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, out_dim)   # → action probs
        self.critic = nn.Linear(hidden_dim, 1)         # → V(s)
    
    def forward(self, x):
        x = F.relu(self.shared(x))
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value

if __name__ == "__main__":
    env_name = "CartPole-v1"

    env = gym.make(env_name, render_mode="None")
    obs, info = env.reset(seed=SEED)

    total_reward = 0
    ep = 0
    ep_rewards = [0]
    steps = 100000 * 1

    model = ActorCritic(in_dim = 4, out_dim = 2)
    optim   = torch.optim.Adam(model.parameters(), lr=1e-3)

    gamma = 0.99

    acts = []
    rewards = []
    obs_s = []
    log_probs_old = []
    values_list = []
    dones = []

    K_epochs = 8
    clip_eps = 0.2
    batch_size = 64
    lr = 3e-4

    for step in range(steps):
        obs = torch.tensor(obs)
        probs, value = model(obs)

        dist = torch.distributions.Categorical(probs)
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        action = action_tensor.item()

        new_obs, reward, terminated, truncated, info = env.step(action)

        obs_s.append(obs)
        acts.append(action_tensor)
        log_probs_old.append(log_prob.detach())
        values_list.append(value.detach())
        rewards.append(reward)
        dones.append(terminated)

        total_reward += reward

        obs = new_obs

        if terminated or truncated:
            obs, info = env.reset()

            # Convert to tensors
            obs_batch = torch.stack(obs_s)
            acts_batch = torch.stack(acts)
            old_log_probs = torch.stack(log_probs_old)
            old_values = torch.stack(values_list).squeeze()

            # Compute returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            
            # (Normalised) Advantages
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update — multiple epochs over same data
            entropy_coef = 0.01  # Coefficient to weight the entropy term
            for epoch in range(K_epochs):
                probs, values = model(obs_batch)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(acts_batch)

                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs) # new_log_probs / old_log_probs
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

                actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                critic_loss = F.mse_loss(values.squeeze(), returns)

                loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

                optim.zero_grad()
                loss.backward()
                optim.step()

            obs_s = []
            acts = []
            log_probs_old = []
            values_list = []
            rewards = []
            dones = []

            if ep % 1000 == 0:
                print(ep, total_reward, np.max(ep_rewards), np.mean(ep_rewards))
            ep += 1
            
            ep_rewards.append(total_reward)
            total_reward = 0

    env.close()

    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset(seed=SEED)

    total_reward = 0
    for steps in range(500):
        obs = torch.tensor(obs)
        probs, value = model(obs)
        action = torch.argmax(probs).item()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()
            print(ep, total_reward)
            total_reward = 0

    plt.plot(ep_rewards)
    plt.show()
    env.close()