import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class Bandit:
    def __init__(self, n_timesteps, n_arms):
        self.arms = list(range(n_arms))
        self.q_true = np.random.normal(loc=0, scale=1, size=10)
        self.rewards = [np.random.normal(q, 1, n_timesteps) for q in self.q_true]

    def reward(self, arm, timestep):
        return self.rewards[arm][timestep]


def simulate(n_tries=2000, n_timesteps=1000, n_arms=10, e=0.0):
    rewards = np.zeros((n_tries, n_timesteps))
    is_optimal = np.zeros((n_tries, n_timesteps))
    optimal_rewards = []
    for t in tqdm(range(n_tries)):
        bandit = Bandit(n_timesteps, n_arms)
        q_estimated = np.zeros(n_arms)
        action_rewards = [[] for _ in range(n_arms)]
        optimal_action = np.argmax(bandit.q_true)
        optimal_rewards.append(np.max(bandit.q_true))
        for i in range(n_timesteps):
            if np.random.rand() < e:
                action = np.random.choice(bandit.arms)
            else:
                max_val = np.max(q_estimated)
                action = np.random.choice(np.where(q_estimated == max_val)[0])
            r = bandit.reward(action, i)
            action_rewards[action].append(r)
            q_estimated[action] = np.mean(action_rewards[action])
            rewards[t][i] = r
            if action == optimal_action:
                is_optimal[t][i] = 1
    print(np.mean(optimal_rewards))
    return rewards, is_optimal


def fig_2_2():
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    epsilons = [0, 0.01, 0.1]
    for e in epsilons:
        result, is_optimal = simulate(e=e)
        rewards_mean = np.mean(result, axis=0)
        optimal_selection = np.mean(is_optimal, axis=0)
        ax[0].plot(rewards_mean, label=f'epsilon = {e}')
        ax[1].plot(optimal_selection, label=f'epsilon = {e}')
    ax[0].legend()
    ax[1].legend()
    # plt.show()
    plt.savefig('images/fig_2_2.png')

def run(k):
    fig_2_2()


if __name__ == '__main__':
    num_bandits = 10
    run(num_bandits)
