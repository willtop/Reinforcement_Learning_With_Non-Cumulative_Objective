# Evaluate script

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import adhoc_wireless_net
import agent
from system_parameters import *
import argparse
    

METHODS = ['Q-Min', 'Q-Sum', 'Q-MC']#, 'Closest to Destination', 'Best Direction']
N_ROUNDS = 2
linestyles = {"Q-Min": "r", "Q-Sum": "b--", "Q-MC": "g-.", "Closest to Destination": "k:", "Best Direction": "y-."}

def method_caller(agent, method, visualize_axis=None):
    if method in ['Q-Min', 'Q-Sum', 'Q-MC']:
        agent.route_DDRQN(visualize_axis)
    elif method == 'DDQN Lowest Interference':
        agent.route_DDRQN_with_lowest_interference_band()
    elif method == 'Strongest Neighbor':
        agent.route_strongest_neighbor()
    elif method == 'Closest to Destination':
        agent.route_close_neighbor_closest_to_destination()
    elif method == 'Least Interfered':
        agent.route_close_neighbor_under_lowest_power()
    elif method == 'Largest Data Rate':
        agent.route_close_neighbor_with_largest_forward_rate()
    elif method == 'Best Direction':
        agent.route_close_neighbor_best_forwarding_direction()
    elif method == 'Destination Directly':
        agent.route_destination_directly()
    else:
        print("Shouldn't be here!")
        exit(1)
    return

# Perform a number of rounds of sequential routing
def sequential_routing(agents, method):
    # 1st round routing, just with normal order
    for agent in agents:
        assert len(agent.flow.get_links()) == 0, "Sequential routing should operate on fresh starts!"
        while not agent.flow.destination_reached():
            method_caller(agent, method)
    # compute bottleneck SINR to determine the routing for the sequential rounds
    for i in range(N_ROUNDS-1):
        bottleneck_rates = []
        for agent in agents:
            agent.process_links(memory=None)
            bottleneck_rates.append(agent.flow.bottleneck_rate)
        ordering = np.argsort(bottleneck_rates)[::-1]
        for agent_id in ordering:  # new round routing
            agent = agents[agent_id]
            agent.reset()
            while not agent.flow.destination_reached():
                method_caller(agent, method)
    for agent in agents:
        agent.process_links(memory=None)
    return

def evaluate_routing(adhocnet, agents, method, n_layouts):
    assert adhocnet.n_flows == len(agents)
    results = []
    for i in range(n_layouts):
        results_oneLayout = []
        adhocnet.update_layout()
        sequential_routing(agents, method)
        for agent in agents:
            results_oneLayout.append([agent.flow.bottleneck_rate, len(agent.flow.get_links()), agent.flow.get_number_of_reprobes()])
        for agent in agents:
            agent.reset()
        results.append(results_oneLayout)
    results = np.array(results); assert np.shape(results)==(n_layouts, adhocnet.n_flows, 3)
    return results

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result', help='whether to plot the results of full evaluations', default=False)
    parser.add_argument('--visualize', help='option to visualize the routing results by all methods', default=False)
    parser.add_argument('--step', help='option to visualize step selection and scores', default=False)
    args = parser.parse_args()
    adhocnet = adhoc_wireless_net.AdHoc_Wireless_Net()
    agents_Q_Min = [agent.Agent(adhocnet, i, "Q-Min") for i in range(adhocnet.n_flows)]
    agents_Q_Sum = [agent.Agent(adhocnet, i, "Q-Sum") for i in range(adhocnet.n_flows)]
    agents_Q_MC = [agent.Agent(adhocnet, i, "Q-MC") for i in range(adhocnet.n_flows)]

    N_LAYOUTS_TEST = 1000
    if args.visualize:
        N_LAYOUTS_TEST = 1

    if (not args.result) and (not args.visualize) and (not args.step):
        all_results = dict()
        for method in METHODS:
            print("Evaluating {}...".format(method))
            if method == "Q-Sum":
                agents =  agents_Q_Sum
            elif method == "Q-MC":
                agents = agents_Q_MC 
            else:
                agents = agents_Q_Min
            all_results[method] = evaluate_routing(adhocnet, agents, method, N_LAYOUTS_TEST)
        # save the results as a dictionary
        with open("DQN/Trained_Models/eval_results.pkl", "wb") as f:
            pickle.dump(all_results, f)
        print(f"Finished evaluation with {N_LAYOUTS_TEST} layouts and results saved!")
    elif args.result:
        with open("DQN/Trained_Models/eval_results.pkl", "rb") as f:
            all_results = pickle.load(f)
        # plot Sum-Rate and Min-Rate CDF curve
        plt.xlabel("Bottleneck Rate (Mbps)", fontsize=25)
        plt.ylabel("Cumulative Distribution over Test Ad hoc Networks", fontsize=25)
        plt.grid(linestyle="dotted")
        plot_upperbound = 0
        for i, (method, results) in enumerate(all_results.items()):
            assert np.shape(results) == (N_LAYOUTS_TEST, adhocnet.n_flows, 3)
            bottleneck_rates, n_links, n_reprobes = results[:,:,0], results[:,:,1], results[:,:,2]
            bottleneck_rates_layoutmean = np.mean(bottleneck_rates, axis=1)
            print("[{}] Avg Bottleneck Rate Layout Mean: {:.3g}Mbps; Std: {:.3g}Mbps; Avg links per flow: {:.1f}; Avg reprobes per flow: {:.2g}".format(
                method, np.mean(bottleneck_rates_layoutmean)/1e6, np.std(bottleneck_rates_layoutmean)/1e6, 
                np.mean(n_links), np.mean(n_reprobes)))
            plt.plot(np.sort(bottleneck_rates.flatten())/1e6, np.arange(1, N_LAYOUTS_TEST*adhocnet.n_flows+1) / (N_LAYOUTS_TEST*adhocnet.n_flows), linestyles[method], linewidth=1.3, label=method)
            plot_upperbound = max(np.max(bottleneck_rates)/1e6, plot_upperbound)
        plt.xlim(left=0, right=0.45*plot_upperbound)
        plt.ylim(bottom=0, top=1)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size':25})
        plt.subplots_adjust(left=0.06,right=0.97,bottom=0.08,top=0.95)
        plt.show()
    elif args.visualize:
        METHODS = ["Q-Min", "Q-Sum", "Q-MC"]
        for i in range(N_LAYOUTS_TEST):
            fig, axes = plt.subplots(1, 3)
            axes = axes.flatten()
            gs = gridspec.GridSpec(1,3)
            gs.update(wspace=0.05, hspace=0.05)
            for (j, method) in enumerate(METHODS):
                if method == "Q-Sum":
                    agents =  agents_Q_Sum
                elif method == "Q-MC":
                    agents = agents_Q_MC 
                else:
                    agents = agents_Q_Min
                ax = axes[j]
                ax.set_title(method, fontsize=25)
                ax.tick_params(axis=u'both', which=u'both', length=0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                sequential_routing(agents, method)
                # start visualization plot
                adhocnet.visualize_layout(ax)
                for agent in agents:
                    agent.visualize_route(ax)
                    agent.reset()
            plt.subplots_adjust(left=0.01,right=0.99,bottom=0.08,top=0.95, wspace=0.05)
            plt.show()
            adhocnet.update_layout()
    elif args.step:
        METHODS = ["Q-Min", "Closest to Destination"]
        for method in METHODS:
            if method == "Q-Sum":
                agents =  agents_Q_Sum
            elif method == "Q-MC":
                agents = agents_Q_MC 
            else:
                agents = agents_Q_Min
            for i, agent in enumerate(agents):
                print("[Sequential Routing 1st round] {} Agent {}".format(method, i))
                while not agent.flow.destination_reached():
                    ax = plt.gca()
                    adhocnet.visualize_layout(ax)
                    for agent_finished in agents[:i]:
                        agent_finished.visualize_route(ax)
                    # execute one step and plot
                    method_caller(agent, method, ax)
                    if agent.flow.destination_reached():
                        agent.visualize_route(ax)
                    plt.tight_layout()
                    plt.show()
            for i, agent in enumerate(agents):
                print("[Sequential Routing 2nd round] {} Agent {}".format(method, i))
                agent.reset()
                while not agent.flow.destination_reached():
                    ax = plt.gca()
                    adhocnet.visualize_layout(ax)
                    # For unrouted agents in this round, also visualize agents' routes in the first round
                    for agent_finished in agents: 
                        if agent_finished == agent:
                            continue
                        agent_finished.visualize_route(ax)
                    # execute one step and plot
                    method_caller(agent, method, ax)
                    if agent.flow.destination_reached():
                        agent.visualize_route(ax)
                    plt.tight_layout()
                    plt.show()
            for agent in agents:
                agent.reset()

    print("Evaluation Completed!")
