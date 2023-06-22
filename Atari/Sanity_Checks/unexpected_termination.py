import random
import os
import numpy as np
import torch
import atari_py
from atari_py import ALEInterface
# from ale_py import ALEInterface
# Set random seed
RANDOM_SEED = 1234
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

ale = ALEInterface()
ale.setInt("random_seed", RANDOM_SEED)
ale.setFloat("repeat_action_probability", 0)
# ale.setBool("display_screen", True)
ale.setString("record_screen_dir", "frames")
ale.loadROM(atari_py.get_game_path("space_invaders"))
# ale.loadROM("space_invaders.bin")
env_actions = ale.getMinimalActionSet()

actions = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2, 2, 2, 4, 2, 2, 2, 2, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 5, 2, 1, 1, 5, 4, 1, 1, 4, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 4, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 5, 5, 5, 1, 5, 5, 5, 1, 5, 5, 5, 1, 5, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 5, 5, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 2, 0, 2, 2, 2, 1, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 4, 4, 2, 2, 2, 2, 5, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 4, 4, 4, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 1, 2, 1, 4, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 1, 5, 5, 5, 5, 5, 5, 3, 3, 2, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 5, 2, 5, 4, 4, 4, 4, 4, 4, 3, 4, 4, 3, 5, 3, 2, 2, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 2, 4, 4, 4, 2, 0, 4, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 1, 5, 5, 5, 1, 2, 4, 5, 2, 0, 0]
frames = 0
score = 0
# start environment with a number of no-op actions
for i in range(random.randrange(30)):
    ale.act(0)  # assume action indexed by 0 is no-op
    if ale.game_over():
        ale.reset_game()

lives = ale.lives()
print("Starting. Lives: ", lives)
for action in actions:
    for i in range(4):
        score += ale.act(env_actions[action])
    if ale.game_over():
        print("Terminate frame at: ", frames*4)
        break
    if lives > ale.lives():
        print("Frames: {}, Score: {}, Lives: {}".format(frames, score, ale.lives()))
        lives = ale.lives()
        # one op action after one life is done
        ale.act(0)
    frames += 1

print("[Final] Frames: {}, Score: {}, Lives: {}".format(frames, score, ale.lives()))
