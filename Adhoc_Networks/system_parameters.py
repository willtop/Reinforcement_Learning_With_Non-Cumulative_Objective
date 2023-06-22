# All environmental numerical settings
import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# wireless network parameters
BANDWIDTH = 5e6
CARRIER_FREQUENCY = 2.4e9
_NOISE_dBm_perHz = -130
NOISE_POWER = np.power(10, ((_NOISE_dBm_perHz-30)/10)) * BANDWIDTH
TX_HEIGHT = 1.5
RX_HEIGHT = 1.5
_TX_POWER_dBm = 30
TX_POWER = np.power(10, (_TX_POWER_dBm - 30) / 10)
_ANTENNA_GAIN_dB = 2.5
ANTENNA_GAIN = np.power(10, (_ANTENNA_GAIN_dB/10))

# Set random seed
RANDOM_SEED = 234
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
