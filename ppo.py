import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers import RecordEpisodeStatistics
from packaging import version
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--num-episodes", type=int, default=10,
        help="the number of episodes to run")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--num_updates", type=int, default=5,
        help="the number of update in each episode")
    parser.add_argument("--clip", type=float, default=0.2,
        help="the value of clip")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
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
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1), std=1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,env.action_space.n), std=0.01),
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
        return self.critic(obs).squeeze(), probs.log_prob(acts)

class PPO():
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.actor_optim = torch.optim.Adam(agent.actor.parameters(), lr=args.learning_rate)
        self.critic_optim = torch.optim.Adam(agent.critic.parameters(), lr=args.learning_rate)
    
    def roll_out(self):
        obs = []
        rews = []
        acts = []
        log_probs = []
        lens = []
        rtgs = []

        observation = self.env.reset()[0]
        done = False 
        ep_rews = []
        for _ in range(args.num_steps):
            obs.append(observation)
            action, log_prob = agent.get_action_and_prob(observation)
            observation, reward, done, truncated, info = env.step(action)
            acts.append(action) #self.acts = np.append(self.acts, action)# 
            log_probs.append(log_prob)
            ep_rews.append(reward)
            if done:
                break
        obs = np.array(obs).reshape(-1, 4)
        log_probs = torch.FloatTensor(log_probs)
        obs = torch.FloatTensor(obs)
        acts = torch.FloatTensor(np.array(acts))
        lens.append(info["episode"]["l"])
        rews.append(ep_rews)
        rtgs = torch.FloatTensor(self.to_go(rews))  # rtgs: nd_array, log_probs: torch_tensor
        return obs, acts, rtgs, log_probs

    def to_go(self, rewards):
        rtgs = []
        disc = 0
        for ep_rew in reversed(rewards):
            disc = 0
            for rew in reversed(ep_rew):
                disc = rew + args.gamma * disc
                rtgs.insert(0, disc)
        return np.array(rtgs)
    
    def advantage_estimate(self, obs, rtgs):
        values = []
        values = agent.get_value(obs)
        advantages = rtgs - values
        advantage = (advantages - advantages.mean())/(advantages.std() - 1e-10)
        return advantages
    
    def train(self):
        for _ in range(args.num_episodes):
            obs, acts, rtgs, log_probs = self.roll_out()
            advantages = self.advantage_estimate(obs, rtgs)
            for _ in range(args.num_updates):
                V, curr_log_probs = agent.evaluate(obs, acts)
                pi_ratios = torch.exp(curr_log_probs - log_probs)
                surrogate1 = pi_ratios*advantages
                surrogate2 = torch.clamp(pi_ratios, 1-args.clip, 1+args.clip)*advantages
                actor_loss = (-torch.min(surrogate1, surrogate2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                loss = nn.MSELoss()
                critic_loss = loss(V, rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = gym.make(args.gym_id)
    env = RecordEpisodeStatistics(env)
    agent = Agent(env)
    ppo = PPO(env)
    ppo.train()
            

