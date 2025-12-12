#!/usr/bin/env python3
import argparse
import math
import random
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

def plot_durations(episode_durations: List[int], show_result: bool = False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float32)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)

def create_bins_and_q_table(obs_space_size, n_actions, num_bins=10):
    bins = [
        np.linspace(-4.8, 4.8, num_bins),
        np.linspace(-4.0, 4.0, num_bins),
        np.linspace(-0.418, 0.418, num_bins),
        np.linspace(-4.0, 4.0, num_bins),
    ]
    q_table = np.random.uniform(
        low=-2.0,
        high=0.0,
        size=([num_bins] * obs_space_size + [n_actions]),
    )
    return bins, q_table

def get_discrete_state(state, bins):
    state = np.clip(
        state,
        [-4.8, -4.0, -0.418, -4.0],
        [ 4.8,  4.0,  0.418,  4.0],
    )
    return tuple(np.digitize(state[i], bins[i]) - 1 for i in range(len(bins)))

def run_tabular(episodes):
    env = gym.make("CartPole-v1")
    obs_space_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    epsilon = 1.0
    end_decay = episodes // 2
    epsilon_decay = epsilon / max(1, end_decay)

    bins, q_table = create_bins_and_q_table(obs_space_size, n_actions)
    episode_lengths = np.zeros(episodes, dtype=np.int32)

    metrics = {"ep": [], "avg": [], "min": [], "max": []}

    for ep in range(episodes):
        state, _ = env.reset()
        discrete_state = get_discrete_state(state, bins)
        done = False
        steps = 0

        while not done:
            steps += 1
            if np.random.rand() > epsilon:
                action = int(np.argmax(q_table[discrete_state]))
            else:
                action = int(np.random.randint(n_actions))

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_discrete_state = get_discrete_state(new_state, bins)

            current_q = q_table[discrete_state + (action,)]
            if done and steps < 200:
                new_q = -375.0
            else:
                max_future_q = np.max(q_table[new_discrete_state])
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                    reward + DISCOUNT * max_future_q
                )

            q_table[discrete_state + (action,)] = new_q
            discrete_state = new_discrete_state

        episode_lengths[ep] = steps

        if ep < end_decay:
            epsilon -= epsilon_decay

        if ep % 100 == 0 and ep > 0:
            recent = episode_lengths[ep - 100 : ep]
            metrics["ep"].append(ep)
            metrics["avg"].append(np.mean(recent))
            metrics["min"].append(np.min(recent))
            metrics["max"].append(np.max(recent))
            print(f"[TABULAR] {ep} {metrics['avg'][-1]:.1f}")

    plt.figure()
    plt.plot(metrics["ep"], metrics["avg"], label="average")
    plt.plot(metrics["ep"], metrics["min"], label="min")
    plt.plot(metrics["ep"], metrics["max"], label="max")
    plt.legend()
    plt.show()
    env.close()

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden=128):
        super().__init__()
        self.l1 = nn.Linear(n_observations, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, n_actions)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

def run_dqn(episodes):
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0
    episode_durations = []

    def select_action(state):
        nonlocal steps_done
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
        steps_done += 1
        if torch.rand(1).item() > eps:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        return torch.tensor([[env.action_space.sample()]], device=device)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        expected = reward_batch + GAMMA * next_state_values

        loss = nn.SmoothL1Loss()(state_action_values, expected.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    plt.ion()
    for i_episode in range(episodes):
        obs, _ = env.reset()
        state = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        for t in count():
            action = select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            reward_t = torch.tensor([reward], device=device)
            next_state = None if done else torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            memory.push(state, action, next_state, reward_t)
            state = next_state
            optimize_model()

            target_sd = target_net.state_dict()
            policy_sd = policy_net.state_dict()
            for k in policy_sd:
                target_sd[k] = policy_sd[k] * TAU + target_sd[k] * (1 - TAU)
            target_net.load_state_dict(target_sd)

            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

    plot_durations(episode_durations, True)
    plt.ioff()
    plt.show()
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["tabular", "dqn"], default="dqn")
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()

    if args.algo == "tabular":
        run_tabular(args.episodes or 20000)
    else:
        run_dqn(args.episodes or (1000 if torch.cuda.is_available() else 600))

if __name__ == "__main__":
    main()
