# Try after loading the ROM and started the game. Would toggling
# use_SDL work or not.
from ale_py import ALEInterface
from ale_py.roms import Pong, Breakout, SpaceInvaders, Kangaroo
import matplotlib.pyplot as plt
import numpy as np

print("Set up an ale")
ale = ALEInterface()
ale.setBool("display_screen", True)
ale.loadROM(Kangaroo)
# ale.loadROM(SpaceInvaders)
actions = ale.getMinimalActionSet()

for i in range(1):
    ale.reset_game()
    total_reward = 0
    while not ale.game_over():
        action = np.random.choice(actions)
        reward = ale.act(action)
        total_reward += reward
        print("[{}] Action: {}; Reward: {}; Total Reward: {}; Life Remaining: {}; All Lives Done: {}".format(i, action, reward, total_reward, ale.lives(), ale.game_over()))
