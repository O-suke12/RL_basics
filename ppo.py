import argparse
import datetime
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gym.wrappers import RecordEpisodeStatistics
from packaging import version
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment"
    )
    parser.add_argument("--gym-id", type=str, default="CartPole-v1", help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=4000, help="the number of episodes to run")
    parser.add_argument(
        "--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--num-updates", type=int, default=4, help="the number of update in each episode")
    parser.add_argument("--clip", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument(
        "--render", type=lambda x: bool(strtobool(x)), default=False, help="if toggled, render the environment"
    )
    parser.add_argument("--num-minibatches", type=int, default=6)
    args = parser.parse_args()
    return args


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, std)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        x = x.clone().detach()
        return self.critic(x).squeeze()

    def get_action_and_prob(self, x, action=None):
        logits = self.actor(torch.from_numpy(x.astype(np.float32)))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action.numpy(), probs.log_prob(action)

    def evaluate(self, obs, acts):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        return self.critic(obs).squeeze(), probs.log_prob(acts),  probs.entropy()


class PPO:
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.actor_optim = torch.optim.Adam(agent.actor.parameters(), lr=args.learning_rate)
        self.critic_optim = torch.optim.Adam(agent.critic.parameters(), lr=args.learning_rate)

    def roll_out(self):
        obs = []
        acts = []
        log_probs = []
        lens = []
        rtgs = []
        ep_rews = []

        observation = self.env.reset()[0]
        done = False
        for _ in range(args.num_steps):
            obs.append(observation)
            action, log_prob = agent.get_action_and_prob(observation)
            observation, reward, done, truncated, info = env.step(action)
            acts.append(action) 
            log_probs.append(log_prob)
            ep_rews.append(reward)
            if done:
                break
        obs = np.array(obs).reshape(-1, 4)
        log_probs = torch.FloatTensor(log_probs)
        obs = torch.FloatTensor(obs)
        acts = torch.FloatTensor(np.array(acts))
        rtgs = torch.FloatTensor(self.to_go(ep_rews)) 
        return (
            obs,
            acts,
            rtgs,
            log_probs,
        )

    def to_go(self, rewards):
        rtgs = []
        disc = 0
        for rew in reversed(rewards):
            disc = rew + args.gamma * disc
            rtgs.insert(0, disc)
        return np.array(rtgs)

    def advantage_estimate(self, obs, rtgs):
        values = []
        values = agent.get_value(obs)
        advantages = rtgs - values
        advantage = (advantages - advantages.mean()) / (advantages.std() - 1e-10)
        return advantages

    def train(self):
        total_upd = args.num_episodes*args.num_updates
        for epi in range(args.num_episodes):
            if epi%100==0:
                print(f"{epi}\n")
            obs, acts, rtgs, log_probs = self.roll_out()
            advantages = self.advantage_estimate(obs, rtgs)
            advantages = advantages.detach()

            step = obs.size(0)
            inds = np.arange(step)
            minibatch_size = step//args.num_minibatches
            for upd in range(args.num_updates):
                sofar_upd = epi * args.num_updates + upd

                #learning rate annealing which doesn't work well but it is good with mini-batch
                frac = (sofar_upd - 1.0) / total_upd
                new_lr = args.learning_rate*(1.0-frac)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr

                #mini-batch works significantly well
                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = obs[idx]
                    mini_acts = acts[idx]
                    mini_log_probs = log_probs[idx]
                    mini_advantages = advantages[idx]
                    mini_rtgs = rtgs[idx]
                    V, curr_log_probs, entropy = agent.evaluate(mini_obs, mini_acts)
                    pi_ratios = torch.exp(curr_log_probs - mini_log_probs)
                    surrogate1 = pi_ratios * mini_advantages
                    surrogate2 = torch.clamp(pi_ratios, 1 - args.clip, 1 + args.clip) * mini_advantages
                    actor_loss = (-torch.min(surrogate1, surrogate2)).mean()
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    loss = nn.MSELoss()
                    critic_loss = loss(V, mini_rtgs)
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

                writer.add_scalar("loss-actor", actor_loss, sofar_upd)
                writer.add_scalar("loss-critic", critic_loss, sofar_upd)
            writer.add_scalar("epi-length", len(rtgs), sofar_upd)


if __name__ == "__main__":
    args = parse_args()
    now = datetime.datetime.now()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{now.strftime('%Y-%m-%d--%H-%M')}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = gym.make(args.gym_id, render_mode="rgb_array")
    env = RecordEpisodeStatistics(env)
    if args.render:
        env = gym.wrappers.RecordVideo(env, "./video", episode_trigger = lambda x: x % 100 == 0)
    agent = Agent(env)
    ppo = PPO(env)
    ppo.train()
    env.close()

    
