# Standard DQN Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *

class DQN_Atari(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=HISTORY_LENGTH, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.n_conv_out_features = 3136
        self.fc_1 = nn.Linear(self.n_conv_out_features, 512)
        self.fc_2 = nn.Linear(512, n_actions)

    def forward(self, x):
        minibatch_size, n_in_channels, height, width = x.size()
        assert n_in_channels == HISTORY_LENGTH and height == width == 84
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = x.view(-1, self.n_conv_out_features)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

class DQN_Gym(nn.Module):
    def __init__(self, n_actions, n_state_dims):
        super().__init__()
        self.fc_1 = nn.Linear(n_state_dims, 100)
        self.fc_2 = nn.Linear(100, 100)
        self.fc_3 = nn.Linear(100, 100)
        self.fc_4 = nn.Linear(100, n_actions)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_4(x)
        return x

