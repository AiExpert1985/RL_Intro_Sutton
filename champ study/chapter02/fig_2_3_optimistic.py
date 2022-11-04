# created 3-Nov-2022
# Book: Reinforcement Learning - an introduction
# by Richard Sutton - 2nd edition
# chapter 02 - section 2.6
# optimistic initial values
# for stationary bandit problem


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


def simulate(n_bandits=2000, n_timesteps=1000, n_arms=10, e=0.0, alpha=0.1, init_type="realistic"):
    reward_grid = np.zeros((n_bandits, n_timesteps))
    optimality_grid = np.zeros((n_bandits, n_timesteps))
    for b in tqdm(range(n_bandits)):
        bandit = Bandit(n_timesteps, n_arms)
        Q = np.ones(n_arms) * 5 if init_type == 'optimistic' else np.zeros(n_arms)
        for t in range(n_timesteps):
            optimal_action = np.argmax(bandit.q)
            if np.random.rand() < e:
                action = np.random.choice(bandit.arms)
            else:
                max_val = np.max(Q)
                action = np.random.choice(np.where(Q == max_val)[0])
            reward = bandit.reward(action, t)
            Q[action] = Q[action] + alpha * (reward - Q[action])
            reward_grid[b][t] = reward
            if action == optimal_action:
                optimality_grid[b][t] = 1
    return reward_grid, optimality_grid


def fig_2_3():
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    epsilons = [0, 0.1]
    init_types = ["optimistic", "realistic"]
    for e, i in zip(epsilons, init_types):
        rewards, optimality = simulate(e=e, init_type=i)
        rewards_mean = np.mean(rewards, axis=0)
        optimal_selection = np.mean(optimality, axis=0)
        ax[0].plot(rewards_mean, label=f'{i}, epsilon = {e}')
        ax[1].plot(optimal_selection, label=f'{i}, epsilon = {e}')
    ax[0].legend()
    ax[1].legend()
    # plt.show()
    plt.savefig('images/fig_2_3.jpg')

def run(k):
    fig_2_3()


if __name__ == '__main__':
    num_bandits = 10
    run(num_bandits)
