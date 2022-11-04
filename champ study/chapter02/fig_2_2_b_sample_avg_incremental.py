# created 3-Nov-2022
# Book: Reinforcement Learning - an introduction
# by Richard Sutton - 2nd edition
# chapter 02 - section 2.4
# incremental implementation for sample averages method, with e-greedy
# comparing results of :
# (1) greedy
# (2) e-greedy with epsilon = 0.1
# (3) e-greedy with epsilon = 0.01


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class Bandit:
    def __init__(self, n_timesteps, n_arms):
        self.arms = list(range(n_arms))
        self.q = np.random.normal(loc=0, scale=1, size=10)
        self.rewards = [np.random.normal(q, 1, n_timesteps) for q in self.q]

    def reward(self, arm, timestep):
        return self.rewards[arm][timestep]


def simulate(n_bandits=2000, n_timesteps=1000, n_arms=10, e=0.0):
    reward_grid = np.zeros((n_bandits, n_timesteps))
    optimality_grid = np.zeros((n_bandits, n_timesteps))
    for b in tqdm(range(n_bandits)):
        bandit = Bandit(n_timesteps, n_arms)
        Q = np.zeros(n_arms)
        N = np.zeros(n_arms)
        for t in range(n_timesteps):
            optimal_action = np.argmax(bandit.q)
            if np.random.rand() < e:
                action = np.random.choice(bandit.arms)
            else:
                max_val = np.max(Q)
                action = np.random.choice(np.where(Q == max_val)[0])
            N[action] += 1
            reward = bandit.reward(action, t)
            Q[action] = Q[action] + 1/N[action] * (reward - Q[action])
            reward_grid[b][t] = reward
            if action == optimal_action:
                optimality_grid[b][t] = 1
    return reward_grid, optimality_grid


def fig_2_2_b():
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    epsilons = [0, 0.01, 0.1]
    for e in epsilons:
        rewards, optimality = simulate(e=e)
        rewards_mean = np.mean(rewards, axis=0)
        optimal_selection = np.mean(optimality, axis=0)
        ax[0].plot(rewards_mean, label=f'epsilon = {e}')
        ax[1].plot(optimal_selection, label=f'epsilon = {e}')
    ax[0].legend()
    ax[1].legend()
    # plt.show()
    plt.savefig('images/fig_2_2_b_incremental.png')

def run(k):
    fig_2_2_b()


if __name__ == '__main__':
    num_bandits = 10
    run(num_bandits)
