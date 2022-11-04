import numpy as np
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter

DECK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 10, 'Q': 10, 'K': 10, 'A': 11}


HIT = 0
STAND = 1
ACTIONS = [HIT, STAND]

N_EPISODES = 100000

GAMMA = 0.9
ALPHA = 0.1

EPSILON_FINAL = 0.01
EPSILON_START = 1.0
EPSILON_DECAY = N_EPISODES / 2

TERMINAL_STATE = (0, False, 0)
state_action_value = defaultdict(lambda: [0, 0])
state_visit = defaultdict(lambda: [0, 0])


def pick_card():
    return np.random.choice(DECK)


def evaluate_card(card):
    return CARD_VALUE[card]


def evaluate_hand(cards_sum, usable_ace, card):
    cards_sum += evaluate_card(card)
    if cards_sum > 21 and (usable_ace or card == 'A'):
        cards_sum -= 10
        usable_ace = False
    if usable_ace and card == 'A':
        usable_ace = True
    return cards_sum, usable_ace


def initialize_game():
    usable_ace = False
    player_cards = [pick_card() for _ in range(2)]
    if 'A' in player_cards:
        usable_ace = True
    player_sum = np.sum([evaluate_card(card) for card in player_cards])
    # if player has 2 aces, use one of them (i.e. consider one of them as 1 instead of 11)
    if player_sum == 22:
        player_sum = 12
    dealer_card_val = CARD_VALUE[pick_card()]
    initial_state = (player_sum, usable_ace, dealer_card_val)
    return initial_state


def player_fixed_policy(state):
    player_sum, usable_ace, dealer_card_val = state
    action = STAND if player_sum >= 15 else HIT
    return action


def player_ucb_policy(state, t):
    state_action_val = state_action_value[state]
    action_counts = state_visit[state]
    ucb = np.sqrt(np.log(t + 1) / (np.array(action_counts) + 0.00001))
    state_action_val += ucb
    if state_action_val[0] == state_action_val[1]:
        action = np.random.choice(ACTIONS)
    else:
        action = np.argmax(state_action_val)
    return action


def player_e_greedy_policy(state, t):
    state_action_vals = state_action_value[state]
    epsilon = max(EPSILON_FINAL, (EPSILON_START - (t / EPSILON_DECAY)))
    if np.random.rand() < epsilon or state_action_vals[0] == state_action_vals[1]:
        return np.random.choice(ACTIONS)
    action = np.argmax(state_action_vals)
    return action


def dealer_policy(dealer_sum):
    action = STAND if dealer_sum >= 17 else HIT
    return action


def player_turn(state, t):
    trajectory = []
    player_sum, usable_ace, dealer_card_val = state
    while True:
        action = player_ucb_policy(state, t)
        # action = player_e_greedy_policy(state, t)
        if action == STAND:
            trajectory.append((state, action))
            break
        player_card = pick_card()
        player_sum, usable_ace = evaluate_hand(player_sum, usable_ace, player_card)
        if player_sum > 21:
            trajectory.append((state, action))
            break
        next_state = (player_sum, usable_ace, dealer_card_val)
        trajectory.append((state, action))
        state = next_state
    return trajectory, player_sum


def dealer_turn(card_val):
    usable_ace = False
    if card_val == 11:
        usable_ace = True
    dealer_sum = card_val
    while dealer_sum <= 22:
        card_val = pick_card()
        dealer_sum, usable_ace = evaluate_hand(dealer_sum, usable_ace, card_val)
        action = dealer_policy(dealer_sum)
        if action == STAND:
            break
    return dealer_sum


def game_result(player_sum, dealer_sum):
    if player_sum > dealer_sum:
        result = 1
    elif player_sum < dealer_sum:
        result = -1
    else:
        result = 0
    return result


def play_game(t):
    initial_state = initialize_game()
    # initial_state = (21, True, 11)
    trajectory, player_sum = player_turn(initial_state, t)
    if player_sum > 21:
        result = -1
    else:
        dealer_card_val = initial_state[2]
        dealer_sum = dealer_turn(dealer_card_val)
        result = 1 if dealer_sum > 21 else game_result(player_sum, dealer_sum)
    return trajectory, result


def monte_carlo_update(trajectory, result):
    discounts = [1]
    for _ in range(len(trajectory)):
        discounts.append(discounts[-1] * GAMMA)
    discounts.reverse()
    trajectory.reverse()
    for (state, action), discount in zip(trajectory, discounts):
        state_action_vals = state_action_value[state]
        counts = state_visit[state]
        counts[action] += 1
        n = counts[action]
        state_action_vals[action] += 1/n * (discount * result - state_action_vals[action])
        state_action_value[state] = state_action_vals
        state_visit[state] = counts


def monte_carlo_policy_evaluation(trajectories, rewards):
    visited_states = defaultdict(list)
    discounts = [GAMMA ** i for i in range(12)]  # max steps in each game can't exceed 11
    for trajectory, r in zip(trajectories, rewards):
        trajectory.reverse()
        for (s, a), gamma in zip(trajectory, discounts[:len(trajectory)]):
            G = gamma * r
            visited_states[(s, a)].append(G)
    # now update state_values with new values
    for (s, a), val in visited_states.items():
        # print(s, a, "=", val)
        state_action_value[s][a] = np.mean(val)
        state_visit[s][a] += len(val)


def run_simulation(n_games):
    writer = SummaryWriter(comment="-blackjack")
    results = []
    for t in tqdm(range(n_games)):
        trajectory, result = play_game(t)
        monte_carlo_update(trajectory, result)
        results.append(result)
        writer.add_scalar("result_100", np.mean(results[-100:]), t)
    print("player's mean score of last 100 games = ", np.round(np.mean(results[-int(len(results)/2):]), 3))
    print(len(state_action_value))
    for key in sorted(state_action_value.keys(), reverse=True):
        val = state_action_value[key]
        print(f'{key}: [{round(val[0], 2)}, {round(val[1], 2)}]; count = {state_visit[key]}')


def run_eval_improv_simulation(total_games, eval_every_n):
    results = []
    trajectories = []
    for t in tqdm(range(total_games)):
        if t % eval_every_n == 0:
            monte_carlo_policy_evaluation(trajectories, results[-len(trajectories):])
            trajectories = []
        trajectory, result = play_game(t)
        # print(trajectory, result)
        trajectories.append(trajectory)
        results.append(result)
    print("player's mean score of last 100 games = ", np.round(np.mean(results[int(len(results)/2):]), 3))
    print(len(state_action_value))
    for key in sorted(state_action_value.keys(), reverse=True):
        val = state_action_value[key]
        print(f'{key}: [{round(val[0], 2)}, {round(val[1], 2)}]; count = {state_visit[key]}')


if __name__ == '__main__':
    # run_simulation(N_EPISODES)
    run_eval_improv_simulation(N_EPISODES, eval_every_n=10000)
