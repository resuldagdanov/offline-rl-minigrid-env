import gym
import gym_minigrid
import random
import pybullet_envs
import numpy as np
import torch
import argparse
from collections import deque
from cql_agent import CQLAgent


def get_config():
    parser = argparse.ArgumentParser(description='Offline-RL')

    parser.add_argument("--run_name", type=str, default="CQL-DQN", help="Run name, default: CQL-DQN")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes, default: 10")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--is_render", type=int, default=1, help="Render environment during training when set to 1, default: 1")
    parser.add_argument("--model_path", type=str, default="./trained_models/cql-dqn_mini-grid_trained-agent_eps300.pth", help="Directory of the loaded model")
    
    args = parser.parse_args()
    return args


def set_seed(config, env):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    gym.utils.seeding.np_random(config.seed)


def evaluate(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make(config.env)

    # initialize with default seed
    set_seed(config, env)

    steps = 0
    total_steps = 0
    average10 = deque(maxlen=10)
    
    agent = CQLAgent(state_size=env.observation_space['image'].shape, action_size=env.action_space.n, device=device)
    agent.network.load_state_dict(torch.load(config.model_path))
    agent.network.eval()

    for i in range(1, config.episodes + 1):

        # reset seeds in every episode so that the agent starts from a random states
        config.seed = i
        set_seed(config, env)

        state = env.reset()
        
        episode_steps = 0
        rewards = 0.0

        while True:
            action = agent.get_greedy_action(state['image'])
            steps += 1
            
            next_state, reward, done, _ = env.step(action[0])
            
            state = next_state
            rewards += reward
            episode_steps += 1

            if config.is_render:
                env.render()
            
            if done:
                break

        average10.append(rewards)
        total_steps += episode_steps
        
        print("Episode: {} | Reward: {} | Steps: {}".format(i, rewards, steps))

    env.close()


if __name__ == "__main__":
    config = get_config()

    evaluate(config)
