# Set random seed to make example reproducable.
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

states = generate_states()


def generate_states():
    states = []
    for row in range(env.track.shape[0]):
        for col in range(env.track.shape[1]):
            for vel_y in range(-10, 11):
                for vel_x in range(-10, 11):
                    states.append((row, col, vel_y, vel_x))
    return states


def Q_table(key_lst):
    q = {}
    for state in key_lst:
        q[state] = [0] * len(env.ACTIONS_DICT)
    return q


def choose_action(policy, state):
    if np.random.uniform() < 0.1:
        action = np.random.choice(env.get_actions())

    else:
        if policy[state] == [0] * len(env.ACTIONS_DICT):
            action = np.random.choice(env.get_actions())
        else:

            action = np.argmax(policy[state])

    return action


def get_average(policy, num_agents):
    total_reward = 0

    for agent in range(num_agents):
        current_state = env.reset()

        while True:
            action = choose_action(policy, current_state)
            next_state, reward, terminal = env.step(action)
            total_reward += reward
            current_state = next_state
            if terminal == True:
                break

    episode_reward = total_reward / num_agents

    return episode_reward


def sarsa(num_of_episodes, num_agents):
    ### Initialise S

    q_table = Q_table(states)

    avg_lst = []
    avg = get_average(q_table, num_agents)
    # avg_lst.append(avg)
    for episode in range(num_of_episodes):
        if (episode + 1) % 10 == 0:
            print("Episode:", episode + 1)

        current_state = env.reset()
        action = choose_action(q_table, current_state)

        while True:
            #             if episode == 149:
            #                 env.render()
            next_state, reward, terminal = env.step(action)

            new_action = choose_action(q_table, next_state)
            q_table[current_state][action] = q_table[current_state][action] + alpha * (
                        reward + gamma * q_table[next_state][new_action] - q_table[current_state][action])

            action = new_action
            current_state = next_state
            if terminal == True:
                break
        update_avg = get_average(q_table, num_agents)
        avg_lst.append(update_avg)

    return avg_lst


alpha = 0.2
epsilon = 0.1
gamma = 0.9
num_of_episodes = 150
num_agents = 30
sarsa_mean_return = sarsa(num_of_episodes, num_agents)

x_episode = np.linspace(1, num_of_episodes + 1, num_of_episodes)
plt.figure()
plt.plot(x_episode, sarsa_mean_return)
plt.xlabel('Episode')
plt.ylabel('Mean Return')
g_title = 'The Number of Agents: ' + str(num_agents)
plt.title(g_title)