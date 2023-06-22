import numpy as np
from settings import *
from ale_py import ALEInterface
from ale_py.roms import Breakout
import gym
from collections import deque
import cv2
import random

class Environment():
    def __init__(self):
        # Fields child-classes have
        self.env = None
        self.n_actions = None
        self.n_state_dims = None
        self.state = None
        self.all_lives_done = None

    def reset(self):
        pass

    def step(self):
        pass


class Environment_Atari(Environment):
    def __init__(self, use_SDL, store_Frames, seed_ID):
        super(Environment_Atari, self).__init__()
        self.env = ALEInterface()
        self.env.setInt("random_seed", RANDOM_SEEDS[seed_ID])
        self.env.setFloat("repeat_action_probability", 0)  # always repeat action for frame_skip number of frames
        if use_SDL:
            self.env.setBool("display_screen", True)
        if store_Frames:
            self.env.setString("record_screen_dir", "Frames_{}".format(ENVIRONMENT_NAME))
        self.env.setBool('color_averaging', False)
        print(f"Loading Atari ROM for the game: {ENVIRONMENT_NAME}")
        if ENVIRONMENT_NAME == "Breakout":
            self.env.loadROM(Breakout)
        else:
            print('Incorrect environment!')
            exit(1)
        self._env_actions = self.env.getMinimalActionSet()
        self.n_actions = len(self._env_actions)
        self.state = deque(maxlen=HISTORY_LENGTH)
        self.all_lives_done = False
        self._lives = self.env.lives()
        print(f"Environment {ENVIRONMENT_NAME} initialized! Agent with {self._lives} lives")

    # For games like space-invaders, have multiple lives and here requires two reset behaviors
    # For games like pong, have only one life and requires just the total reset behavior
    def reset(self):
        # First of all, clear all state buffer
        for _ in range(HISTORY_LENGTH):
            self.state.append(np.zeros([84,84]))
        # Only reset the game after all agent's lives are done
        if self.all_lives_done:
            self.env.reset_game()
            self._lives = self.env.lives()
            self.all_lives_done = False
            # Perform no-op of a number of steps each time a new episode starts
            for i in range(random.randrange(30)):
                self.env.act(0) # assume action indexed by 0 is no-op
                if self.env.game_over():
                    self.env.reset_game()
        else:
            self.env.act(0) # step one no-op after one life is lost
        # Get the first game screen after game reset or a new agent life starts
        screen = self._get_screen()
        self.state.append(screen)
        return np.array(self.state)

    def _get_screen(self):
        screen = cv2.resize(self.env.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        screen = screen/255
        return screen

    def step(self, action):
        reward = 0
        episode_done = False
        screen_buffer = np.zeros([2, 84, 84])
        for i in range(FRAME_SKIP):
            reward += self.env.act(self._env_actions[action])
            if i == FRAME_SKIP-2:
                screen_buffer[0] = self._get_screen()
            elif i == FRAME_SKIP-1:
                screen_buffer[1] = self._get_screen()
            self.all_lives_done = self.env.game_over()
            if self.all_lives_done:
                episode_done = True
                break
        observation = np.max(screen_buffer, axis=0) # If game ends earlier, gonna have all zero observation, which doesn't matter
        self.state.append(observation)
        # See if there is episodic (single life) ending
        lives = self.env.lives()
        if lives < self._lives:
            # For these games with more than one life but no negative reward in losing life, 
            # manually assign negative reward
            if ENVIRONMENT_NAME == "Breakout":
                reward = -1
            episode_done = True
            self._lives = lives
        return np.array(self.state), reward, episode_done

class Environment_Gym(Environment):
    def __init__(self):
        super(Environment_Gym, self).__init__()
        if ENVIRONMENT_NAME == "CartPole":
            self.env = gym.make("CartPole-v1")
        else:
            print(f"Invalid environment name {ENVIRONMENT_NAME}!")
            exit(1)
        self.n_actions = self.env.action_space.n
        self.n_state_dims = self.env.observation_space.shape[0]
        assert np.shape(self.env.observation_space.sample()) == (self.n_state_dims, )
        self.all_lives_done = False
        # to detect whether the game terminates due to successfully running up to max number of steps
        self.episode_step_counter = 0 

    def reset(self):
        self.episode_step_counter = 0
        self.all_lives_done = False
        return self.env.reset()

    def step(self, action):
        next_state, reward, episode_done, _ = self.env.step(action)
        self.episode_step_counter += 1 
        # For cartpole doesn't differentiate termination in reward
        if episode_done:
            if ENVIRONMENT_NAME == "CartPole" and self.episode_step_counter < self.env._max_episode_steps:
                reward = -1
            if ENVIRONMENT_NAME == "CartPole" and self.episode_step_counter == self.env._max_episode_steps:
                #print(f"Agent reached max steps in an episode in {ENVIRONMENT_NAME}!")
                pass
            self.all_lives_done = True
        return next_state, reward, episode_done

    def render(self):
        self.env.render()
