# Contain tesing script and functions computing all evaluating metrics
import numpy as np
import argparse
from agent import Regular_DDQN_Agent, Modified_DDQN_Agent
from environment import Environment_Atari, Environment_Gym
from settings import *

def compute_Qs_over_random_states(agent, memory_valid):
    states = []
    for i in range(EVALUATION_STATES):
        state, _ = memory_valid.get_current_and_next_state(i)
        states.append(state)
    states = np.array(states) 
    Qs = agent.main_net_predict(states).squeeze().detach().cpu().numpy()
    return np.mean(np.max(Qs, axis=1))

# Expect a separate environment distinct from the environment used for training
def compute_scores(agent, env, n_trails, random=False):
    scores_all_trials = []
    for i in range(n_trails):
        actions = []
        score = 0
        env.all_lives_done = True # ensure environment restart
        state = env.reset()
        n_frames = 0
        while True:
            if n_trails == 1 and ENVIRONMENT_NAME in ["CartPole"]:
                env.render()
            if not random:
                if ENVIRONMENT_NAME in ["CartPole"]:
                    # Pure agent's action for simple openai-gym control task
                    action = agent.act_epsilon_greedy(state, 0)
                else:
                    # Have 0.05 random exploration probability for evaluating the policy, as per DQN paper
                    action = agent.act_epsilon_greedy(state, 0.05)
            else:
                action = agent.act_epsilon_greedy(state, 1.0)
            next_state, reward, episode_done = env.step(action)
            actions.append(action)
            score += reward
            if env.all_lives_done:
                break
            if episode_done:
                state = env.reset()
            else:
                state = next_state
            n_frames += 1
        scores_all_trials.append(score)
    return np.mean(scores_all_trials)


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="evaluate script argument parser")
    parser.add_argument("--visualize", default=False, help="whether visualizing one trial or running full testing")
    args = parser.parse_args()

    # Prepare the environment, agent, and replay_memory
    if ENVIRONMENT_NAME == "CartPole":
        env = Environment_Gym()
    else:        
        env = Environment_Atari(use_SDL=args.visualize, store_Frames=False, seed_ID=0)
    agents = {}
    agents["Regular"] = Regular_DDQN_Agent(n_actions=env.n_actions, n_state_dims=env.n_state_dims, seed_ID=0)
    agents["Modified"] = Modified_DDQN_Agent(n_actions=env.n_actions, n_state_dims=env.n_state_dims, seed_ID=0)

    for agent_type, agent in agents.items():
        print(f"[{agent_type}]")
        avg_score = compute_scores(agent, env, 1 if args.visualize else EVALUATION_TRIALS_TEST)
        print("Score: {}".format(avg_score))
    avg_score = compute_scores(agents["Regular"], env, 1 if args.visualize else EVALUATION_TRIALS_TEST, random=True)
    print("[Random Policy] Score: {}".format(avg_score))

    print("Script Finished Successfully!")
