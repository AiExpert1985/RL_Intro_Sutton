# created 4-Nov-2022
# Book: Reinforcement Learning - an introduction
# by Richard Sutton - 2nd edition
# chapter 02 - section 2.8
# implementing Gradient Bandit Algorithms
# for stationary bandit problem


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class Bandit:
    def __init__(self, n_arms, n_timesteps, q_base):
        self.n_arms = n_arms
        self.arms = list(range(n_arms))
        self.q = np.random.normal(loc=q_base, scale=1, size=10)
        self.rewards = [np.random.normal(q, 1, n_timesteps) for q in self.q]

    def reward(self, arm, timestep):
        return self.rewards[arm][timestep]


def simulate(n_bandits=200, n_timesteps=1000, n_arms=10, alpha=0.1, base="no baseline"):
    q_base = 0 if base == "no baseline" else 5
    reward_grid = np.zeros((n_bandits, n_timesteps))
    optimality_grid = np.zeros((n_bandits, n_timesteps))
    for b in tqdm(range(n_bandits)):
        bandit = Bandit(n_arms, n_timesteps, q_base)
        optimal_action = np.argmax(bandit.q)
        H = np.zeros(n_arms)
        reward_mean = 0
        for t in range(n_timesteps):
            H_exp = np.exp(H)
            Pi = H_exp / np.sum(H_exp)
            action = np.random.choice(bandit.arms, p=Pi)
            reward = bandit.reward(action, t)
            reward_mean += (reward - reward_mean) / (t + 1)
            mask = np.zeros(bandit.n_arms)
            mask[action] = 1
            H += alpha * (reward - reward_mean) * (mask - Pi)
            reward_grid[b][t] = reward
            optimality_grid[b][t] = int(action == optimal_action)
    return reward_grid, optimality_grid


def fig_2_3():
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    alpha = [0.1, 0.4]
    base = ["baseline", "no baseline"]
    for b in base:
        for a in alpha:
            rewards, optimality = simulate(alpha=a, base=b)
            rewards_mean = np.mean(rewards, axis=0)
            optimal_selection = np.mean(optimality, axis=0)
            ax[0].plot(rewards_mean, label=f'{b}, alpha={a}')
            ax[1].plot(optimal_selection, label=f'{b}, alpha={a}')
    ax[0].legend()
    ax[1].legend()
    plt.savefig('images/fig_2_5.jpg')

def run(k):
    fig_2_3()


if __name__ == '__main__':
    num_bandits = 10
    run(num_bandits)
