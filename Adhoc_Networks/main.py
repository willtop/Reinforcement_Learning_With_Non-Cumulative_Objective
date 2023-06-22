# Main script to train the agent

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import trange
import sys
import argparse
import evaluate
sys.path.append("DQN/")
import adhoc_wireless_net
import agent
from replay_memory import Prioritized_Replay_Memory, Uniform_Replay_Memory
from system_parameters import *
    

INITIAL_EXPLORE_LAYOUTS = int(30e3)
EPSILON_GREEDY_LAYOUTS = int(300e3)
FINAL_CONVERGE_LAYOUTS = int(50e3)
REPLAY_MEMORY_SIZE = int(100e3)
EVALUATE_FREQUENCY = int(2e3)
TARGET_NET_SYNC_FREQUENCY = int(5e3)

# Linear scheduler for hyper-parameters during training. Modified from the code from openai baselines:
#     https://github.com/openai/baselines/tree/master/baselines
class LinearSchedule():
    def __init__(self, initial_val, final_val, schedule_timesteps):
        self.initial_val = initial_val
        self.final_val = final_val
        self.schedule_timesteps = schedule_timesteps
    def value(self, timestep):
        fraction = max(min(float(timestep) / self.schedule_timesteps, 1.0),0.0)
        return self.initial_val + fraction * (self.final_val - self.initial_val)

def save_training_metrics(metrics_all):
    for key, val in metrics_all.items():
        np.save("DQN/Trained_Models/training_metrics_DDQN_{}.npy".format(key), np.array(val))
        n_steps = len(val)
    print("Stored training metrics after {} training steps".format(n_steps))
    return    

def smooth_curve(raw_data):
    AVERAGE_WINDOW = 10
    n_data = np.size(raw_data)
    assert n_data > AVERAGE_WINDOW and np.shape(raw_data) == (n_data, )
    data_smoothed = []
    # firstly, compute the full convolutions with average window size
    data_smoothed_tmp = np.convolve(raw_data, np.ones(AVERAGE_WINDOW)/AVERAGE_WINDOW, mode='valid')
    assert np.shape(data_smoothed_tmp) == (n_data-AVERAGE_WINDOW+1, )
    # then compute the initial AVERAGE_WINDOW-1 steps reward average
    data_smoothed_prepend = []
    for j in range(1, AVERAGE_WINDOW):
        data_smoothed_prepend.append(np.sum(raw_data[:j])/j)
    data_smoothed = np.concatenate([data_smoothed_prepend, data_smoothed_tmp],axis=0)
    assert np.shape(data_smoothed)==(n_data,)
    return data_smoothed

def plot_training_curves():
    linestyles = {"Q-Min": "r", "Q-Sum": "b--", "Q-MC": "g-."}
    linestyles_raw = {"Q-Min": "r:", "Q-Sum": "b:", "Q-MC": "g:"}
    print("Plot training curves...")
    fig, axes = plt.subplots(1,2)
    axes[0].set_xlabel("DQN Weight Update Steps (1e3)", fontsize=25)
    axes[0].set_ylabel("Q Loss (log-scale)", fontsize=25)
    axes[1].set_xlabel("DQN Weight Update Steps (1e3)", fontsize=25)
    axes[1].set_ylabel("Avg Bottleneck Rate (Mbps)", fontsize=25)
    for reward_type in ["Q-Min", "Q-MC", "Q-Sum"]:
        val = np.load("DQN/Trained_Models/training_metrics_DDQN_{}.npy".format(reward_type))
        xs = val[:,0]/1e3
        mse, objval = val[:,1], val[:,2]
        # plot the raw data curve
        axes[0].semilogy(xs, mse, linestyles_raw[reward_type], linewidth=0.07)
        axes[1].plot(xs, objval, linestyles_raw[reward_type], linewidth=0.07)
        # plot the smoothed data curve
        mse_smoothed, objval_smoothed = smooth_curve(mse), smooth_curve(objval)
        axes[0].semilogy(xs, mse_smoothed, linestyles[reward_type], linewidth=2.2, label=reward_type)
        axes[1].plot(xs, objval_smoothed, linestyles[reward_type], linewidth=2.2, label=reward_type)
    axes[0].legend(prop={"size":22})
    axes[1].legend(prop={"size":22})
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    plt.subplots_adjust(left=0.06,right=0.96,bottom=0.11,top=0.95)
    plt.show()
    return

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="main script arguments")
    parser.add_argument("--plot", default=False, help="whether to plot the training curves (execute after training)")
    args = parser.parse_args()
    if args.plot:
        plot_training_curves()
        exit(0)

    adhocnet = adhoc_wireless_net.AdHoc_Wireless_Net()
    # Train all types of agents together: Q-MC, Q-Min and Q-Sum
    agents_all = {}
    metrics_all = {} # store two metrics Q-Loss, routing rate performance
    best_avg_bottleneck_rates_all = {}
    memories_all = {}
    for reward_type in ["Q-Min", "Q-MC", "Q-Sum"]:
        agents_all[reward_type] = [agent.Agent(adhocnet, i, reward_type) for i in range(adhocnet.n_flows)]
        metrics_all[reward_type] = []
        best_avg_bottleneck_rates_all[reward_type] = -np.inf
        memories_all[reward_type] = Uniform_Replay_Memory(REPLAY_MEMORY_SIZE)

    policy_epsilon = LinearSchedule(initial_val=1.0, final_val=0.1, schedule_timesteps=EPSILON_GREEDY_LAYOUTS)
    for i in trange(1, INITIAL_EXPLORE_LAYOUTS+EPSILON_GREEDY_LAYOUTS+FINAL_CONVERGE_LAYOUTS+1):
        adhocnet.update_layout() # refresh to a newly generated the layout, shared among agents with all reward types
        for reward_type, agents in agents_all.items():
            # Firstly, complete the other data flows with a reasonable benchmark
            for agent in agents[:-1]:
                while not agent.flow.destination_reached():
                    agent.route_close_neighbor_closest_to_destination()
            # Route the remaining data flow with epsilon-greedy policy
            while not agents[-1].flow.destination_reached():
                # set the final settlement stage with zero exploration probability for all reward types
                epsilon_val = 0 if i>(INITIAL_EXPLORE_LAYOUTS+EPSILON_GREEDY_LAYOUTS) else policy_epsilon.value(i-1-INITIAL_EXPLORE_LAYOUTS)
                agents[-1].route_epsilon_greedy(epsilon=epsilon_val)
            agents[-1].process_links(memories_all[reward_type])
            for agent in agents:
                agent.reset()
            if i >= INITIAL_EXPLORE_LAYOUTS:  # Have gathered enough experiences, start training the agents
                Q_loss = agents[-1].train(memories_all[reward_type])
                assert not (np.isnan(Q_loss) or np.isinf(Q_loss)) 
                if (i % TARGET_NET_SYNC_FREQUENCY == 0):
                    agents[-1].sync_target_network()
                if (i % EVALUATE_FREQUENCY == 0):
                    for agent in agents[:-1]: # load the currently trained model parameters to evaluate
                        agent.sync_main_network_from_another_agent(agents[-1])
                    eval_results = evaluate.evaluate_routing(adhocnet, agents, "DDQN_{}".format(reward_type), n_layouts=100)
                    avg_bottleneck_rate = np.mean(eval_results[:,:,0])/1e6
                    metrics_all[reward_type].append([i, Q_loss, avg_bottleneck_rate])
                    if best_avg_bottleneck_rates_all[reward_type] < avg_bottleneck_rate:
                        agents[-1].save_dqn_model()
                        best_avg_bottleneck_rates_all[reward_type] = avg_bottleneck_rate
                    save_training_metrics(metrics_all)

    print("<<<<<<<<<<<<<<<<<<<<Training Complete!>>>>>>>>>>>>>>>>>>>>>>>")
    save_training_metrics(metrics_all)
    
    for key, agents in agents_all.items():
        agents[-1].visualize_non_zero_rewards(memories_all[key])    

    print("Script Finished Successfully!")
