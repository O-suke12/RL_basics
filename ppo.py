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
    parser.add_argument("--num-episodes", type=int, default=1000, help="the number of episodes to run")
    parser.add_argument(
        "--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--num-updates", type=int, default=4, help="the number of update in each episode")
    parser.add_argument("--clip", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument(
        "--wandb-project-name", type=str, default="ppo-implementation-details", help="the wandb's project name"
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--render",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, render the environment",
    )
    parser.add_argument("--num-minibatches", type=int, default=6)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--lamda", type=float, default=0.95)
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
        return self.critic(obs).squeeze(), probs.log_prob(acts), probs.entropy()


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
        ep_vals = []
        ep_dones = []
        t = 0

        observation = self.env.reset()[0]
        done = False
        for _ in range(args.num_steps):
            t += 1
            obs.append(observation)
            action, log_prob = agent.get_action_and_prob(observation)
            observation, reward, done, truncated, info = env.step(action)

            log_probs.append(log_prob)
            ep_dones.append(done)
            ob_val = np.array(observation).reshape(-1, 4)
            ob_val = torch.FloatTensor(ob_val)
            val = agent.get_value(ob_val)

            acts.append(action)
            ep_rews.append(reward)
            ep_vals.append(float(val))
            if done:
                break

        obs = np.array(obs).reshape(-1, 4)
        log_probs = torch.FloatTensor(log_probs)
        obs = torch.FloatTensor(obs)
        acts = torch.FloatTensor(np.array(acts))
        rtgs = torch.FloatTensor(self.to_go(ep_rews))
        return (obs, acts, rtgs, log_probs, ep_rews, ep_dones, ep_vals)

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

    def calclulate_gae(self, rews, values, dones):
        batch_advantages = []
        advantages = []
        last_advantage = 0

        for t in reversed(range(len(rews))):
            if t + 1 < len(rews):
                delta = rews[t] + args.gamma * values[t + 1] * (1 - dones[t + 1]) - values[t]
            else:
                delta = rews[t] - values[t]
            advantage = delta + args.gamma * args.lamda * last_advantage * (1 - dones[t])
            last_advantage = advantage
            advantages.insert(0, advantage)
        batch_advantages.extend(advantages)
        return torch.tensor(batch_advantages, dtype=torch.float32)

    def train(self):
        total_upd = args.num_episodes * args.num_updates
        for epi in range(args.num_episodes):
            if epi % 100 == 0:
                print(f"{epi}\n")
            obs, acts, rtgs, log_probs, rews, dones, vals = self.roll_out()
            advantages = self.calclulate_gae(rews, vals, dones)
            # advantages = self.advantage_estimate(obs, rtgs)
            advantages = advantages.detach()

            step = obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // args.num_minibatches
            for upd in range(args.num_updates):
                sofar_upd = epi * args.num_updates + upd

                # learning rate annealing which doesn't work well but it is good with mini-batch
                frac = (sofar_upd - 1.0) / total_upd
                new_lr = args.learning_rate * (1.0 - frac)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr

                # mini-batch
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
                    entropy = entropy.mean()
                    log_ratios = curr_log_probs - mini_log_probs
                    pi_ratios = torch.exp(curr_log_probs - mini_log_probs)
                    approx_kl = ((1 - pi_ratios) - log_ratios).mean()
                    surrogate1 = pi_ratios * mini_advantages
                    surrogate2 = torch.clamp(pi_ratios, 1 - args.clip, 1 + args.clip) * mini_advantages
                    actor_loss = (-torch.min(surrogate1, surrogate2)).mean() - args.entropy_coeff * entropy
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                    self.actor_optim.step()

                    loss = nn.MSELoss()
                    critic_loss = loss(V, mini_rtgs)
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                    self.critic_optim.step()
                    if approx_kl > args.target_kl:
                        break

                writer.add_scalar("loss-actor", actor_loss, sofar_upd)
                writer.add_scalar("loss-critic", critic_loss, sofar_upd)
            writer.add_scalar("epi-length", len(rtgs), sofar_upd)


if __name__ == "__main__":
    args = parse_args()
    now = datetime.datetime.now()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{now.strftime('%Y-%m-%d--%H-%M')}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = gym.make(args.gym_id, render_mode="rgb_array")
    env = RecordEpisodeStatistics(env)
    if args.render:
        env = gym.wrappers.RecordVideo(env, "./video", episode_trigger=lambda x: x % 100 == 0)
    agent = Agent(env)
    ppo = PPO(env)
    ppo.train()
    env.close()
