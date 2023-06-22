# The main script to train the agent for a given model

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import trange
import argparse
from settings import *
from agent import Regular_DDQN_Agent, Modified_DDQN_Agent
from environment import Environment_Atari, Environment_Gym
from replay_memory import *
from evaluate import compute_Qs_over_random_states, compute_scores


# Linear scheduler for hyper-parameters during training.
# Modified on top of the code from openai baselines:
#     https://github.com/openai/baselines/tree/master/baselines
class LinearSchedule():
    def __init__(self, initial_val, final_val, schedule_timesteps):
        self.initial_val = initial_val
        self.final_val = final_val
        self.schedule_timesteps = schedule_timesteps
    def value(self, timestep):
        fraction = max(min(float(timestep) / self.schedule_timesteps, 1.0), 0.0)
        return self.initial_val + fraction * (self.final_val - self.initial_val)
    
def train(seed_id):
    # Prepare a pair of environments, agents, and replay_memories
    envs_train, memories_train = {}, {}
    for agent_type in ["Regular", "Modified"]:
        if ENVIRONMENT_NAME == 'CartPole':
            envs_train[agent_type] = Environment_Gym()
            if MEMORY_TYPE == "Prioritized":
                memories_train[agent_type] = Prioritized_Replay_Memory_Gym(REPLAY_MEMORY_SIZE)
            else:
                memories_train[agent_type] = Uniform_Replay_Memory_Gym(REPLAY_MEMORY_SIZE)
        else:
            envs_train[agent_type] = Environment_Atari(use_SDL=False, store_Frames=False, seed_ID=seed_id)
            if MEMORY_TYPE == "Prioritized":
                memories_train[agent_type] = Prioritized_Replay_Memory_Atari(REPLAY_MEMORY_SIZE)
            else:
                memories_train[agent_type] = Uniform_Replay_Memory_Atari(REPLAY_MEMORY_SIZE)
    if ENVIRONMENT_NAME == 'CartPole':
        env_valid = Environment_Gym()
        if MEMORY_TYPE == "Prioritized":
            memory_valid = Prioritized_Replay_Memory_Gym(EVALUATION_STATES)
        else:
            memory_valid = Uniform_Replay_Memory_Gym(EVALUATION_STATES)
    else:
        env_valid = Environment_Atari(use_SDL=False, store_Frames=False, seed_ID=seed_id)
        if MEMORY_TYPE == "Prioritized":
            memory_valid = Prioritized_Replay_Memory_Atari(EVALUATION_STATES)
        else:
            memory_valid = Uniform_Replay_Memory_Atari(EVALUATION_STATES)
    agents = {}
    agents["Regular"] = Regular_DDQN_Agent(n_actions=envs_train["Regular"].n_actions, n_state_dims=envs_train["Regular"].n_state_dims, seed_ID=seed_id)
    agents["Modified"] = Modified_DDQN_Agent(n_actions=envs_train["Modified"].n_actions, n_state_dims=envs_train["Regular"].n_state_dims, seed_ID=seed_id)


    print("[{}] Generate validation states via random acting...".format(ENVIRONMENT_NAME))
    state = env_valid.reset()
    for i in range(EVALUATION_STATES):
        action = agents["Regular"].act_epsilon_greedy(state, 1.0)
        next_state, _, episode_done = env_valid.step(action)
        memory_valid.add(state, 1, 0.0, episode_done)
        if episode_done:
            state = env_valid.reset()
        else:
            state = next_state

    print("[{}] Start training...".format(ENVIRONMENT_NAME))
    states, metrics, highest_scores = {}, {}, {}
    for agent_type in agents.keys():
        states[agent_type] = envs_train[agent_type].reset()
        metrics[agent_type] = []
        highest_scores[agent_type] = -np.inf
    # According to Double-DQN paper, epsilon greedy policy has epsilon reaches its lowest value in 1M steps
    policy_epsilon = LinearSchedule(initial_val=1.0, final_val=0.1, schedule_timesteps=1e6) 
    # If prioritize replay is used
    priority_ImpSamp_beta = LinearSchedule(initial_val=0.4, final_val=1.0, schedule_timesteps=TRAIN_STEPS)
    # Firstly, evaluate and record model performance prior to any training
    for agent_type, agent in agents.items():
        agent.eval_mode()
        avg_Q = compute_Qs_over_random_states(agent, memory_valid)
        avg_score = compute_scores(agent, env_valid, EVALUATION_TRIALS)
        metrics[agent_type].append([0, 0, avg_Q, avg_score])
        print("Initial: [{}] Q: {}; Score: {}; historic highest score: {}".format(agent_type, avg_Q, avg_score, highest_scores[agent_type]))  
        if avg_score >= highest_scores[agent_type]:
            print(f"[{agent_type}] Reached highest score!")
            highest_scores[agent_type] = avg_score
    for i in trange(1, INITIAL_EXPLORE_STEPS+TRAIN_STEPS+1):
        # Train parallely over two agent types on each step
        for agent_type, agent in agents.items():
            agent.train_mode()
            action = agent.act_epsilon_greedy(states[agent_type], policy_epsilon.value(i-1-INITIAL_EXPLORE_STEPS))
            next_state, reward, episode_done = envs_train[agent_type].step(action)
            memories_train[agent_type].add(states[agent_type], action, reward, episode_done)
            if episode_done:
                states[agent_type] = envs_train[agent_type].reset()
            else:
                states[agent_type] = next_state
            if (i<=INITIAL_EXPLORE_STEPS):
                continue
            # Train
            if ((i-INITIAL_EXPLORE_STEPS)%TRAIN_FREQUENCY == 0):
                loss = agent.train(memories_train[agent_type], priority_ImpSamp_beta.value(i-i-INITIAL_EXPLORE_STEPS))
            if ((i-INITIAL_EXPLORE_STEPS)%TARGET_NET_SYNC_FREQUENCY == 0):
                agent.sync_target_network()
            # Validation
            if ((i-INITIAL_EXPLORE_STEPS)%EVALUATION_FREQUENCY==0):
                agent.eval_mode()
                avg_Q = compute_Qs_over_random_states(agent, memory_valid)
                avg_score = compute_scores(agent, env_valid, EVALUATION_TRIALS)
                metrics[agent_type].append([int((i-INITIAL_EXPLORE_STEPS)/TRAIN_FREQUENCY), loss, avg_Q, avg_score])
                print("[{}] Q_loss: {}; Q: {}; Score: {}; historic highest score: {}".format(agent_type, loss, avg_Q, avg_score, highest_scores[agent_type]))  
                if avg_score >= highest_scores[agent_type]:
                    print(f"[{agent_type}] Reached highest score!")
                    highest_scores[agent_type] = avg_score
                # always save model to avoid not saving at later training due to earlier noisy evaluation peak
                agent.save_trained_net()

    print(f"##############Finished training on {ENVIRONMENT_NAME} at seed id {seed_id}############")
    # put all the metrics into numpy array for ease of further processing
    for agent_type, vals in metrics.items():
        metrics[agent_type] = np.array(vals) 
    return metrics

if (__name__ == "__main__"):
    metrics_all_seeds = {}
    for i, rand_seed in enumerate(RANDOM_SEEDS):
        set_random_seed(rand_seed)
        metrics = train(seed_id=i)
        metrics_all_seeds[i] = metrics
        # save metrics once after each seed is finished
        with open(f"Models/{ENVIRONMENT_NAME}/{ENVIRONMENT_NAME}_metrics_all_seeds.pkl", "wb") as f:
            pickle.dump(metrics_all_seeds, f)

    print("Script finished!")