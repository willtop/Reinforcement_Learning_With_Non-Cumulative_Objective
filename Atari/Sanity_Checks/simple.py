import atari_py
import cv2
import matplotlib.pyplot as plt
import numpy as np

ale = atari_py.ALEInterface()
ale.loadROM(atari_py.get_game_path("space_invaders"))
actions = ale.getMinimalActionSet()

rewards = []
for i in range(10):
    ale.reset_game()
    while not ale.game_over():
        action = np.random.choice(actions)
        reward = ale.act(action)
        rewards.append(reward)
        print("[{}] Action: {}; Reward: {}; Life Remaining: {}".format(i, action, reward, ale.lives()))
print("Rewards: max: {}, min: {}, mean: {}".format(np.max(rewards),np.min(rewards),np.mean(rewards)))
plt.hist(rewards)
plt.show()

print("Script finished successfully")
