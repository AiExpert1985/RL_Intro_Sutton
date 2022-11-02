import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


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

def main(k):
    fig_2_1()


if __name__ == '__main__':
    num_bandits = 10
    main(num_bandits)
