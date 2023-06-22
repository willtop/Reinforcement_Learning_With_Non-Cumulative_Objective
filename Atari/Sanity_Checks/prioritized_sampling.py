import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import sys
sys.path.append("../")
from replay_memory import Prioritized_Replay_Memory
from settings import *

N_SAMPLES = int(1e4)

if(__name__=="__main__"):
    mem = Prioritized_Replay_Memory(N_SAMPLES)
    for i in trange(1,N_SAMPLES+1):
        state = np.ones([HISTORY_LENGTH,84,84])*i
        action = i
        reward = i
        episode_done = (i%2==0)
        mem.add(state, action, reward, episode_done)

    mem.update_priorities(np.arange(N_SAMPLES), np.arange(N_SAMPLES)**(1/0.6))

    actions_sampled = []
    importance_weights_sampled = []
    for i in trange(500):
        _, actions, _, _, _, importance_weights, _ = mem.sample(beta=0.4)
        for action, importance_weight in zip(actions, importance_weights):
            actions_sampled.append(action)
            importance_weights_sampled.append(importance_weight)
    plt.hist(actions_sampled, bins=N_SAMPLES)
    plt.show()
    plt.hist(importance_weights_sampled, bins=100)
    plt.show()