import sys
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA = 1.0
LR = 1e-2
MAX_T = 1000
TRAINING_EPISODES = 500
EVAL_EPISODES = 10

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fcl1 = nn.Linear(s_size, h_size)
        self.fcl2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fcl1(x))
        return F.softmax(self.fcl2(x), dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def Reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []
    history = []

    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]
        
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        for t in range(n_steps)[::1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma*disc_return_t+rewards[t])

        eps = np.finfo(np.float32).eps.item()  #due to numerical instabilities

        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/(returns.std()+eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob*disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))
    return np.array(scores)

def evaluate(env, max_steps, n_eval_episode, policy):
    episode_rewards = []
    for ep in range(n_eval_episode):
        rewards = []
        done = False
        total_reward_ep = 0
        state = env.reset()[0]

        for i in range(0, max_steps):
            action, _ = policy.act(state)
            state, reward, done, terminated, _ = env.step(action)
            total_reward_ep += reward
            
            if done or terminated:
                break
       
        episode_rewards.append(total_reward_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def main(mode):
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    policy = Policy(env.observation_space.shape[0], env.action_space.n, 16)

    if mode == "train":
        optimizer = torch.optim.Adam(policy.parameters(), LR)
        scores = Reinforce(env, policy, optimizer, TRAINING_EPISODES, MAX_T, GAMMA, True)
        torch.save(policy.state_dict(), "REINFORCE.pth")
        plt.plot (scores)
        plt.savefig('../outputs/REINFORCE_train.png')
    elif mode == "test":
        policy.load_state_dict(torch.load('REINFORCE.pth'))   
        print(evaluate(env, MAX_T, EVAL_EPISODES, policy))
    else:
        print("Please enter a valid mode name")


if __name__ == "__main__":
    main(sys.argv[1])
