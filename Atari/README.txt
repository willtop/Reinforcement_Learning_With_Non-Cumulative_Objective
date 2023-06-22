This folder is for the AAAI main track submission "Control and Reinforcement Learning for Non-Cumulative Objective Functions".
It includes all code files for CartPole and Breakout simulations.

Python Scripts:
  main.py --- the training script to train agents for all explored reinforcement learning methods.
  evaluate.py --- the evaluation script to evaluate the trained agents.
  settings.py --- the script to specify on the task (either CartPole or Breakout) and hyperparameters used.
  agent.py --- the implementation of Q-Min and Q-Sum algorithms, along with other fundamental functionalities in reinforcement learning agents.

Trained Models:
  Trained Models for Q-Min and Q-Sum algorithms under all random seeds are stored under the folders Models/CartPole and Models/Breakout respectively. 
  The learning metrics during training are also stored in these folders as pickle files.