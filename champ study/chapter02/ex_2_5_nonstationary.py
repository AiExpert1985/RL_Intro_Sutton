# 3-Nov-2022
# Book: Reinforcement Learning - an introduction
# by Richard Sutton - 2nd edition
# chapter 02 - section 2.5
# solution for exercise 2.5


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class NonStationaryBandit:
    def __init__(self, n_arms):
        self.arms = list(range(n_arms))
        initial_q = np.random.normal(loc=0, scale=1)
        self.q = np.ones(n_arms) * initial_q

    def step(self, arm):
        self.q += np.random.normal(loc=0, scale=0.01, size=len(self.q))
        return self.reward(arm)

    def reward(self, arm):
        return np.random.normal(loc=self.q[arm], scale=1)


def simulate(n_bandits=2000, n_timesteps=10000, n_arms=10, e=0.1, alpha_type='constant', alpha=0.1):
    reward_grid = np.zeros((n_bandits, n_timesteps))
    optimality_grid = np.zeros((n_bandits, n_timesteps))
    for b in tqdm(range(n_bandits)):
        bandit = NonStationaryBandit(n_arms)
        N = np.zeros(n_arms)
        Q = np.zeros(n_arms)
        for t in range(n_timesteps):
            optimal_action = np.argmax(bandit.q)  # because it is changed at every timestep
            if np.random.rand() < e:
                action = np.random.choice(bandit.arms)
            else:
                max_val = np.max(Q)
                action = np.random.choice(np.where(Q == max_val)[0])
            N[action] += 1
            reward = bandit.step(action)
            if alpha_type != "constant":
                alpha = 1/N[action]
            Q[action] = Q[action] + alpha * (reward - Q[action])
            reward_grid[b][t] = reward
            if action == optimal_action:
                optimality_grid[b][t] = 1
    return reward_grid, optimality_grid


def ex_2_5():
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    alpha_types = ['constant', '1/n']
    for a in alpha_types:
        rewards, optimality = simulate(alpha_type=a)
        rewards_mean = np.mean(rewards, axis=0)
        optimal_selection = np.mean(optimality, axis=0)
        ax[0].plot(rewards_mean, label=f'alpha_type = {a}')
        ax[1].plot(optimal_selection, label=f'alpha_type = {a}')
    ax[0].legend()
    ax[0].set_ylabel("Rewards")
    ax[0].set_xlabel("Timestep")
    ax[1].legend()
    ax[1].set_ylabel("% Optimal action")
    ax[1].set_xlabel("Timestep")
    plt.savefig('images/ex_2_5_nonstationary.png')

def run(k):
    ex_2_5()


if __name__ == '__main__':
    num_bandits = 10
    run(num_bandits)
