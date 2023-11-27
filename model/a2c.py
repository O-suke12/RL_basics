import argparse
import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="Experiment name",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    args = parser.parse_args()
    return args


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )


if __name__ == "__main__":
    args = parse_args()
    writer = SummaryWriter(f"runs/{args.exp_name}")
    x = np.random.randn(100)
    y = x.cumsum()
    for i in range(100):
        writer.add_scalar("x", x[i], i)
        writer.add_scalar("y", y[i], i)

    env = gym.make(args.env)
    agent = Agent(env)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    for i in range(args.num_episodes):
        state = env.reset()
        done = False
        for step in range(args.num_steps):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
               

    writer.close()
