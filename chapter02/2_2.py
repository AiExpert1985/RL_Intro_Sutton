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


def fig_2_1(k=10):
    bandits = np.random.randn(k)
    rewards = [np.random.normal(q_start, 1, 10000) for q_start in bandits]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.violinplot(rewards)
    ax.set_title("10-armed Testbed")
    ax.set_xlabel("Action")
    ax.set_xticks(range(0, 11, 1))
    ax.set_ylabel("Reward Distribution")
    # plt.show()
    plt.savefig('images/fig_2_1.png')


def run(n_tries=2000, n_timesteps=1000, n_arms=10, e=0.0):
    rewards = np.zeros((n_tries, n_timesteps))
    for t in tqdm(range(n_tries)):
        bandit = Bandit(n_timesteps, n_arms)
        q_estimated = np.zeros(n_arms)
        action_rewards = [[] for _ in range(n_arms)]
        for i in range(n_timesteps):
            max_val = np.max(q_estimated)
            if np.random.rand() < e:
                action = np.random.choice(bandit.arms)
            else:
                action = np.random.choice(np.where(q_estimated == max_val)[0])
            r = bandit.reward(action, i)
            action_rewards[action].append(r)
            q_estimated[action] = np.mean(action_rewards[action])
            rewards[t][i] = r
    return rewards


def fig_2_2():
    for e in [0.0, 0.01, 0.1]:
        print('e =', e)
        result = run(e=e)
        rewards_mean = np.mean(result, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(rewards_mean)
    plt.show()

def main(k):
    # fig_2_1()
    fig_2_2()


if __name__ == '__main__':
    num_bandits = 10
    main(num_bandits)
