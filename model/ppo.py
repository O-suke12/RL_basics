# Standard Library
import datetime
import os

# Third Party Library
import gymnasium as gym
import hydra
import numpy as np
import pytz
import torch
import torch.nn as nn
from gymnasium.wrappers import RecordEpisodeStatistics
from omegaconf import DictConfig, omegaconf
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, std)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(env.observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(env.observation_space.shape).prod(), 64)
            ),
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
        return (
            self.critic(obs).squeeze(),
            probs.log_prob(acts),
            probs.entropy(),
        )


class PPO:
    def __init__(
        self, env, writer, device, agent, cfg: DictConfig = DictConfig
    ):
        super().__init__()
        self.env = env
        self.writer = writer
        self.device = device

        self.agent = agent
        self.cfg = cfg
        self.actor_optim = torch.optim.Adam(
            self.agent.actor.parameters(), lr=self.cfg.learning_rate
        )
        self.critic_optim = torch.optim.Adam(
            self.agent.critic.parameters(), lr=self.cfg.learning_rate
        )

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
        for _ in range(self.cfg.num_steps):
            t += 1
            obs.append(observation)
            action, log_prob = agent.get_action_and_prob(observation)
            observation, reward, done, truncated, info = env.step(action)

            log_probs.append(log_prob)
            ep_dones.append(done)
            # ob_val = np.array(observation).reshape(-1, env.observation_space.shape[0])
            ob_val = torch.FloatTensor(observation)
            val = agent.get_value(ob_val)

            acts.append(action)
            ep_rews.append(reward)
            ep_vals.append(float(val))
            if done:
                break

        obs = np.array(obs).reshape(-1, env.observation_space.shape[0])
        log_probs = torch.FloatTensor(log_probs)
        obs = torch.FloatTensor(obs)
        acts = torch.FloatTensor(np.array(acts))
        rtgs = torch.FloatTensor(self.to_go(ep_rews))
        return (obs, acts, rtgs, log_probs, ep_rews, ep_dones, ep_vals)

    def to_go(self, rewards):
        rtgs = []
        disc = 0
        for rew in reversed(rewards):
            disc = rew + self.cfg.gamma * disc
            rtgs.insert(0, disc)
        return np.array(rtgs)

    def advantage_estimate(self, obs, rtgs):
        values = []
        values = agent.get_value(obs)
        advantages = rtgs - values
        advantage = (advantages - advantages.mean()) / (
            advantages.std() - 1e-10
        )
        return advantages

    def calclulate_gae(self, rews, values, dones):
        batch_advantages = []
        advantages = []
        last_advantage = 0

        for t in reversed(range(len(rews))):
            if t + 1 < len(rews):
                delta = (
                    rews[t]
                    + self.cfg.gamma * values[t + 1] * (1 - dones[t + 1])
                    - values[t]
                )
            else:
                delta = rews[t] - values[t]
            advantage = (
                delta
                + self.cfg.gamma
                * self.cfg.lamda
                * last_advantage
                * (1 - dones[t])
            )
            last_advantage = advantage
            advantages.insert(0, advantage)
        batch_advantages.extend(advantages)
        return torch.tensor(batch_advantages, dtype=torch.float32)

    def train(self):
        total_upd = self.cfg.num_episodes * self.cfg.num_updates
        for epi in range(self.cfg.num_episodes):
            if epi % 100 == 0:
                print(f"{epi}\n")
            obs, acts, rtgs, log_probs, rews, dones, vals = self.roll_out()
            # advantages = self.calclulate_gae(rews, vals, dones)
            advantages = self.advantage_estimate(obs, rtgs)
            advantages = advantages.detach()

            step = obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.cfg.num_minibatches
            for upd in range(aself.cfgrgs.num_updates):
                sofar_upd = epi * self.cfg.num_updates + upd

                # learning rate annealing which doesn't work well but it is good with mini-batch
                frac = (sofar_upd - 1.0) / total_upd
                new_lr = self.cfg.learning_rate * (1.0 - frac)
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

                    V, curr_log_probs, entropy = agent.evaluate(
                        mini_obs, mini_acts
                    )
                    entropy = entropy.mean()
                    log_ratios = curr_log_probs - mini_log_probs
                    pi_ratios = torch.exp(curr_log_probs - mini_log_probs)
                    approx_kl = ((1 - pi_ratios) - log_ratios).mean()
                    surrogate1 = pi_ratios * mini_advantages
                    surrogate2 = (
                        torch.clamp(
                            pi_ratios, 1 - self.cfg.clip, 1 + self.cfg.clip
                        )
                        * mini_advantages
                    )
                    actor_loss = (
                        -torch.min(surrogate1, surrogate2)
                    ).mean() - self.cfg.entropy_coeff * entropy
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
                    if approx_kl > self.cfg.target_kl:
                        break

                self.writer.add_scalar("loss-actor", actor_loss, sofar_upd)
                self.writer.add_scalar("loss-critic", critic_loss, sofar_upd)
            self.writer.add_scalar("epi-length", len(rtgs), sofar_upd)


@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.gym_id not in os.listdir("./results/runs"):
        os.mkdir("./results/runs" + cfg.gym_id)
    japan_tz = pytz.timezone("Japan")
    now = datetime.datetime.now(japan_tz)
    run_name = f"{cfg.exp_name}__{now.strftime('%Y-%m-%d--%H-%M')}"
    if cfg.track:
        # Third Party Library
        import wandb

        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=cfg.project_name,
            entity=None,
            sync_tensorboard=True,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir="./results",
        )
    writer = SummaryWriter(f"./results/runs/{cfg.gym_id}/{run_name}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"
    )
    env = gym.make(cfg.gym_id, render_mode="rgb_array")
    env = RecordEpisodeStatistics(env)
    if cfg.render:
        env = gym.wrappers.RecordVideo(
            env, "./results/video", episode_trigger=lambda x: x % 100 == 0
        )
    agent = Agent(env)
    ppo = PPO(env, writer, device, agent, cfg)
    ppo.train()
    env.close()


if __name__ == "__main__":
    main()
