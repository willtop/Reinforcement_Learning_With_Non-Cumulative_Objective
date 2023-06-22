import gym

env = gym.make('CartPole-v0')
print("Environment action set: ", env.action_space)

for i in range(20):
    state = env.reset()
    print(f"[{i+1}/20] Initial state upon reset: ", state)
    for t in range(100):
        env.render()
        observation, reward, done, _ = env.step(env.action_space.sample())
        print(f"Observation: {observation}; Reward: {reward}; Done: {done}")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
