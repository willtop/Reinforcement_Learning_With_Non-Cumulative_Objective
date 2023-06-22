# The main script to train the agent for a given model

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from agent import Agent
from environment import Environment
from replay_memory import Prioritized_Replay_Memory
from tqdm import trange
from evaluate_metrics import compute_Qs_over_random_states, compute_rewards
import argparse

# settings for environment and agent construction
settings = {
    "environment_name": "space_invaders",
    "Evaluation_States": int(1e3),
    "Evaluation_Trials": 15,
    "Frame_Skip": 4
}

if (__name__ == "__main__"):
    # Prepare the environment, agent, and replay_memory
    env = Environment(settings, use_SDL=False)
    agent = Agent(settings, model_type="Double_DQN", n_actions=env.n_actions)
    memory = [] # original DQN paper setting

    metrics = [] # Each element: [episode_ID, MSE, avg_Qs, avg_episodic_rewards]
    env.all_lives_done = True # To ensure restart the environment after validation episodes collection
    state = env.reset()
    episode_done = False
    avg_reward_max = -np.inf
    for i in trange(10):
        agent.train_mode()
        action = agent.act_epsilon_greedy(state, 1)
        next_state, reward, episode_done = env.step(action)
        memory.append([state, next_state])
        if episode_done:
            state = env.reset()
        else:
            state = next_state
        print(np.mean(np.array(memory[i][0])==np.array(memory[i][1])))
        print(hex(id(memory[i][0])), hex(id(memory[i][1])))



    print("Script finished successfully.")

