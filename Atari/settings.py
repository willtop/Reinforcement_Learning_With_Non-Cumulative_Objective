import torch
import os
import numpy as np
import random
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Options: CartPole, Breakout
ENVIRONMENT_NAME = "Breakout"
# Options: Prioritized, Uniform
MEMORY_TYPE = "Prioritized"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parameters for training and replay buffer
HISTORY_LENGTH = 4
FRAME_SKIP = 4 # For space invaders, taking max of last two frames should be not worse than k=3 in DQN paper
TRAIN_FREQUENCY = 4
if ENVIRONMENT_NAME == "Breakout":
    REWARD_DISCOUNT = 0.95
    INITIAL_EXPLORE_STEPS = int(1e4)
    TRAIN_STEPS = int(2e6)
    TARGET_NET_SYNC_FREQUENCY = int(1e4)
    EVALUATION_FREQUENCY = int(5e4)
    EVALUATION_TRIALS = 25
    EVALUATION_TRIALS_TEST = 500
    REPLAY_MEMORY_SIZE = int(5e5)
    RANDOM_SEEDS = [123, 321, 456, 654, 789, 987]
else:
    REWARD_DISCOUNT = 0.97
    INITIAL_EXPLORE_STEPS = int(5e4)
    TRAIN_STEPS = int(6e6)
    TARGET_NET_SYNC_FREQUENCY = int(1e4)
    EVALUATION_FREQUENCY = int(2e5)
    EVALUATION_TRIALS = 5
    EVALUATION_TRIALS_TEST = 100
    REPLAY_MEMORY_SIZE = int(1e6)
    RANDOM_SEEDS = [123, 321, 456]
EVALUATION_STATES = int(2e2)
assert EVALUATION_FREQUENCY % TRAIN_FREQUENCY == 0
LEARNING_RATE = 2e-4 


def set_random_seed(rand_seed):
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    print(f"<<<<<<<<<<<<<<<<<Finished setting random seed at {rand_seed}!>>>>>>>>>>>>>>>")
    return

