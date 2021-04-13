import numpy as np
import random
seed = 5
random.seed(seed)
np.random.seed(seed)

from racetrack_env import RacetrackEnv
from matplotlib import pyplot as plt

# Instantiate environment object.
env = RacetrackEnv()

# Initialise/reset environment.
state = env.reset()


def generate_states():
    states = []
    for row in range(env.track.shape[0]):
        for col in range(env.track.shape[1]):
            for vel_y in range(-10, 11):
                for vel_x in range(-10, 11):
                    states.append((row, col, vel_y, vel_x))
    return states


states = generate_states()


def Q_table(key_lst):
    q = {}
    for state in key_lst:
        q[state] = [0] * len(env.ACTIONS_DICT)
    return q


def epsilon_soft(key_lst):
    s_policy = {}
    for state in key_lst:
        s_policy[state] = [1 / len(env.ACTIONS_DICT)] * len(env.ACTIONS_DICT)
    return s_policy


def play(env, policy):
    episode = []
    current_state = env.reset()
    while True:

        action = int(np.random.choice(len(env.ACTIONS_DICT), 1, p=policy[current_state])[0])
        next_state, reward, terminal = env.step(action)
        episode.append((current_state, action, reward))
        current_state = next_state

        if terminal == True:
            break

    return episode


def on_policy_MC(num_of_episodes, number_agents):
    ### initialise returns, policy, Q_table ###
    q_table = Q_table(states)
    policy_es = epsilon_soft(states)
    returns = {}
    avg_lst = []
    for i in range(num_of_episodes):
        if (i + 1) % 10 == 0:
            print("Episode:", i + 1)

        G = 0
        episode = play(env, policy_es)

        for rev_timestep in reversed(range(len(episode))):

            state, action, reward = episode[rev_timestep]
            state_action = (state, action)
            G = gamma * G + reward

            if not state_action in [(x[0], x[1]) for x in episode[0:rev_timestep]]:
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]

                q_table[state][action] = np.mean(returns[state_action])
                indices = [i for i, x in enumerate(q_table[state]) if x == max(q_table[state])]
                max_Q = random.choice(indices)

                best_A = max_Q

                policy_es[state] = np.ones(len(env.ACTIONS_DICT)) * epsilon / len(env.ACTIONS_DICT)
                policy_es[state][best_A] = 1 - epsilon + (epsilon / len(env.ACTIONS_DICT))

        avg = get_average_MC(policy_es, number_agents)
        avg_lst.append(avg)
    return avg_lst


def get_average_MC(policy, num_agents):
    total_reward = 0

    for agent in range(num_agents):
        current_state = env.reset()
        while True:

            action = int(np.random.choice(len(env.ACTIONS_DICT), 1, p=policy[current_state])[0])
            next_state, reward, terminal = env.step(action)
            total_reward += reward
            current_state = next_state
            if terminal == True:
                break

    episode_reward = total_reward / num_agents

    return episode_reward


num_agent = 30
gamma = 0.9
epsilon = 0.1
num_episodes = 150

MC_mean_return = on_policy_MC(num_of_episodes=num_episodes, number_agents=num_agent)

x_episode = np.linspace(1, num_episodes + 1, num_episodes)
plt.figure()
plt.plot(x_episode, MC_mean_return)
plt.xlabel('Episode')
plt.ylabel('Mean Return')
g_title = 'The Number of Agents: ' + str(num_agent)
plt.title(g_title)
