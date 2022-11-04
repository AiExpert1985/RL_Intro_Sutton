# created 3-Nov-2022
# Book: Reinforcement Learning - an introduction
# by Richard Sutton - 2nd edition
# chapter 02 - section 2.7
# implementing Upper Confidence Bound (UCB)
# for stationary bandit problem


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class Bandit:
    def __init__(self, n_arms, n_timesteps):
        self.arms = list(range(n_arms))
        self.q = np.random.normal(loc=0, scale=1, size=10)
        self.rewards = [np.random.normal(q, 1, n_timesteps) for q in self.q]

    def reward(self, arm, timestep):
        return self.rewards[arm][timestep]


def simulate(n_bandits=2000, n_timesteps=1000, n_arms=10, e=0.1, alpha=0.1, act_type="ucb", c=2.0):
    reward_grid = np.zeros((n_bandits, n_timesteps))
    optimality_grid = np.zeros((n_bandits, n_timesteps))
    for b in tqdm(range(n_bandits)):
        bandit = Bandit(n_arms, n_timesteps)
        optimal_action = np.argmax(bandit.q)
        Q = np.zeros(n_arms)
        N = np.ones(n_arms) * 0.000001
        for t in range(n_timesteps):
            if act_type == "ucb":
                uncertainty = np.log(t+1) / N
                ucb = Q + c * np.sqrt(uncertainty)
                action = np.argmax(ucb)
            else:
                action = np.random.choice(bandit.arms) if np.random.rand() < e else np.argmax(Q)
            N[action] += 1
            reward = bandit.reward(action, t)
            Q[action] = Q[action] + alpha * (reward - Q[action])
            reward_grid[b][t] = reward
            optimality_grid[b][t] = int(action == optimal_action)
    return reward_grid, optimality_grid


def fig_2_3():
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    init_types = ["ucb", "e-greedy"]
    for i in init_types:
        rewards, optimality = simulate(act_type=i)
        rewards_mean = np.mean(rewards, axis=0)
        optimal_selection = np.mean(optimality, axis=0)
        ax[0].plot(rewards_mean, label=f'{i}')
        ax[1].plot(optimal_selection, label=f'{i}')
    ax[0].legend()
    ax[1].legend()
    plt.savefig('images/fig_2_4.jpg')

def run(k):
    fig_2_3()


if __name__ == '__main__':
    num_bandits = 10
    run(num_bandits)
