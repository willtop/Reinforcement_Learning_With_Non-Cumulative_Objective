# DQN Agent: the agent takes in either DQN or its variants

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN_Atari, DQN_Gym
import os
from settings import *

class Agent():
    def __init__(self, n_actions, n_state_dims):
        self.n_actions = n_actions
        self.n_state_dims = n_state_dims
        self.loss_func = nn.MSELoss(reduction='none')
        # self.loss_func = nn.SmoothL1Loss(reduction='mean')
        self.agent_type = None

    def init_neuralNet(self, seed_ID):
        if ENVIRONMENT_NAME in ["CartPole"]:
            self._main_net = DQN_Gym(self.n_actions, self.n_state_dims).to(DEVICE)
            self._target_net = DQN_Gym(self.n_actions, self.n_state_dims).to(DEVICE)            
        else:
            self._main_net = DQN_Atari(self.n_actions).to(DEVICE)
            self._target_net = DQN_Atari(self.n_actions).to(DEVICE)
        self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models/{}/{}_{}_seed_{}.pt".format(ENVIRONMENT_NAME, self.agent_type, ENVIRONMENT_NAME, seed_ID))
        self.optimizer = optim.Adam(self._main_net.parameters(), lr=LEARNING_RATE)
        self._target_net.eval()
        if os.path.exists(self._model_path):
            msg = "Loading trained model from: {}".format(self._model_path)
            self._main_net.load_state_dict(torch.load(self._model_path))
            self._target_net.load_state_dict(torch.load(self._model_path))
        else:
            msg = "Train from scratch."
        print("Initialized {} Agent for {}! {}".format(self.agent_type, ENVIRONMENT_NAME, msg))

    # Don't perform any squeeze for dimension reduction here
    def main_net_predict(self, state_inputs):
        return self._main_net(torch.tensor(state_inputs, dtype=torch.float32).to(DEVICE))

    # Don't perform any squeeze for dimension reduction here
    def target_net_predict(self, state_inputs):
        return self._target_net(torch.tensor(state_inputs, dtype=torch.float32).to(DEVICE))

    def train(self, replay_memory, priority_ImpSamp_beta):
        if MEMORY_TYPE == "Prioritized":
            states, actions, rewards, next_states, episode_dones, importance_weights, idxes = replay_memory.sample(priority_ImpSamp_beta)
        else:
            states, actions, rewards, next_states, episode_dones = replay_memory.sample()
        self.optimizer.zero_grad()
        Qs = self.main_net_predict(states)[np.arange(replay_memory.batch_size), actions]
        next_actions = np.argmax(self.main_net_predict(next_states).detach().cpu().numpy(),axis=1)
        next_Qs = self.target_net_predict(next_states)[np.arange(replay_memory.batch_size), next_actions].detach().cpu().numpy()
        targets = self._compute_Q_targets(rewards, next_Qs, episode_dones)
        losses = self.loss_func(Qs, torch.tensor(targets, dtype=torch.float32).to(DEVICE))
        if MEMORY_TYPE == "Prioritized":
            loss = torch.mean(losses * torch.tensor(importance_weights, dtype=torch.float32).to(DEVICE))
        else:
            loss = torch.mean(losses)
        loss.backward()
        self.optimizer.step()
        if MEMORY_TYPE == "Prioritized":
            # Update prioritizes for these samples based on TD errors
            TD_errors = np.abs(Qs.detach().cpu().numpy() - targets)
            replay_memory.update_priorities(idxes, TD_errors)
        assert not (np.isnan(loss.item()) or np.isinf(loss.item()))
        return torch.mean(losses).item()

    def sync_target_network(self):
        self._target_net.load_state_dict(self._main_net.state_dict())
        return

    def save_trained_net(self):
        torch.save(self._main_net.state_dict(), self._model_path)
        print("{} model saved at {}!".format(self.agent_type, self._model_path)) 
        return

    def _act_by_prediction(self, state):
        Qs = self.main_net_predict(np.expand_dims(state, axis=0)).squeeze().detach().cpu().numpy()
        assert np.shape(Qs) == (self.n_actions, )
        return np.argmax(Qs)

    def _act_random(self):
        return np.random.randint(low=0,high=self.n_actions)
    
    # To be implemented by subclasses
    def _compute_Q_targets(self):
        pass

    def act_epsilon_greedy(self, state, policy_epsilon):
        rand_val = np.random.uniform()
        if rand_val < policy_epsilon:
            action = self._act_random()
        else:
            action = self._act_by_prediction(state)
        return action

    def train_mode(self):
        self._main_net.train()
        return

    def eval_mode(self):
        self._main_net.eval()
        return


class Regular_DDQN_Agent(Agent):
    def __init__(self, n_actions, n_state_dims, seed_ID):
        super().__init__(n_actions, n_state_dims)
        self.agent_type = "Regular"
        self.init_neuralNet(seed_ID)

    # regular Bellman update for computing Q-targets
    def _compute_Q_targets(self, rewards, next_Qs, episode_dones):
        return rewards + REWARD_DISCOUNT * next_Qs * (1-episode_dones.astype(float))
        

class Modified_DDQN_Agent(Agent):
    def __init__(self, n_actions, n_state_dims, seed_ID):
        super().__init__(n_actions, n_state_dims)
        self.agent_type = "Modified"
        self.init_neuralNet(seed_ID)

    # modified Bellman update for min-reward optimization
    def _compute_Q_targets(self, rewards, next_Qs, episode_dones):        
        targets = np.minimum(rewards, REWARD_DISCOUNT * next_Qs * (1-episode_dones.astype(float)))
        targets = targets + np.maximum(rewards*episode_dones.astype(float), 0)
        return targets
