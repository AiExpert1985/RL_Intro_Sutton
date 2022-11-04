import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')


# noinspection PyAttributeOutsideInit
class Bandit:
    def __init__(self, k_arms, epsilon, is_ucb, alpha):
        self.k = k_arms
        self.epsilon = epsilon
        self.is_ucb = is_ucb
        self.alpha = alpha

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimated = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.t = 0

    def act(self):
        if self.is_ucb:
            ucb = self.q_estimated + 2.0 * np.sqrt(np.log(self.t + 1)/(self.action_count + 1e-5))
            return np.argmax(ucb)
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.k))
        # if more than one state has similar value, choose one of them randomly
        q_best = np.max(self.q_estimated)
        return np.random.choice(np.where(self.q_estimated == q_best)[0])

    def step(self, action):
        self.t += 1
        self.action_count[action] += 1
        reward = np.random.randn() + self.q_true[action]
        self.q_estimated[action] += self.alpha * (reward - self.q_estimated[action])
        return reward


def simulate(bandits, runs, time):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_rewards = np.mean(rewards, axis=1)
    mean_best_action_counts = np.mean(best_action_counts, axis=1)
    return mean_best_action_counts, mean_rewards


def fig_2_2():
    is_ucb = [False, True]
    bandits = [Bandit(k_arms=10, epsilon=0.1, is_ucb=u, alpha=0.1) for u in is_ucb]
    best_action_counts, rewards = simulate(bandits, runs=2000, time=1000)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for e, rew in zip(is_ucb, rewards):
        plt.plot(rew, label=e)
    plt.xlabel("time")
    plt.ylabel("rewards mean")
    plt.legend()
    plt.subplot(2, 1, 2)
    for e, a in zip(is_ucb, best_action_counts):
        plt.plot(a, label=e)
    plt.xlabel("time")
    plt.ylabel("% optimal action")
    plt.legend()
    plt.savefig('../images/fig_2_7.png')


if __name__ == '__main__':
    fig_2_2()
