# Script for plotting training curves from different seeds

import numpy as np
import matplotlib.pyplot as plt
import pickle
from settings import *



if (__name__ == "__main__"):
    print("Plotting training curves...")
    with open(f"Models/{ENVIRONMENT_NAME}/{ENVIRONMENT_NAME}_metrics_all_seeds.pkl", "rb") as f:
        metrics_all_seeds = pickle.load(f)
    n_seeds_included = len(metrics_all_seeds.keys())
    # values for x-axis
    x_vals = metrics_all_seeds[0]["Regular"][:, 0]/1e3
    avgrewards_regular = []
    avgrewards_modified = []
    for metrics in metrics_all_seeds.values():
        avgrewards_regular.append(metrics["Regular"][:,3])
        avgrewards_modified.append(metrics["Modified"][:,3])
    avgrewards_regular, avgrewards_modified = np.array(avgrewards_regular), np.array(avgrewards_modified)
    assert np.shape(avgrewards_regular) == np.shape(avgrewards_modified) == (len(RANDOM_SEEDS), len(x_vals))
    plt.xlabel("DQN Weight Update Steps (1e3)", fontsize=25)
    plt.ylabel(f"Average Episodic Reward (over {EVALUATION_TRIALS} episodes)", fontsize=25)
    # plot the mean curve across all seeds
    plt.plot(x_vals, np.mean(avgrewards_regular, axis=0), 'b--', linewidth=1.5, label="Q-Sum")
    plt.plot(x_vals, np.mean(avgrewards_modified, axis=0), 'r', linewidth=1.5, label="Q-Min")
    # plot the shade indicating confidence intervals
    plt.fill_between(x_vals, np.mean(avgrewards_regular, axis=0)-np.std(avgrewards_regular, axis=0),
                             np.mean(avgrewards_regular, axis=0)+np.std(avgrewards_regular, axis=0), 
                             color='b', alpha=0.4)
    plt.fill_between(x_vals, np.mean(avgrewards_modified, axis=0)-np.std(avgrewards_modified, axis=0),
                             np.mean(avgrewards_modified, axis=0)+np.std(avgrewards_modified, axis=0), 
                             color='r', alpha=0.4)
    if ENVIRONMENT_NAME == "CartPole":
        plt.xticks(fontsize=20)
    else:
        plt.xticks(np.linspace(start=x_vals[0], stop=x_vals[-1], num=7, endpoint=True), fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(prop={'size':22})
    plt.subplots_adjust(left=0.065, right=0.97, bottom=0.08, top=0.97)
    plt.show()
    print(f"Finished Plotting, with {n_seeds_included} seeds!")
    