import os
import sys
import gym
import gym_minigrid
import random
import networkx
import pybullet_envs
import numpy as np
import torch
import argparse
import utils
from collections import deque
from construct_graph import build_graph
from hash_table import HashTable

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from cql_dqn.cql_agent import CQLAgent


def get_config():
    parser = argparse.ArgumentParser(description='Offline-RL')

    parser.add_argument("--run_name", type=str, default="cql-dqn", help="Run name, default: CQL-DQN")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=10_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini batch size, default: 32")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 0.01")
    parser.add_argument("--eps_frames", type=int, default=1e3, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e-5")
    parser.add_argument("--is_render", type=int, default=0, help="Render environment during training when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--model_path", type=str, default="./trained_models/cql-dqn_mini-grid_random-agent_eps300.pth", help="Directory of the loaded model")
    parser.add_argument("--is_collect_from_model", type=int, default=0, help="Collect dataset from pre-trained agent model when set to 1, default: 0")
    
    args = parser.parse_args()
    return args


def make_env():
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    env = gym.make(config.env)
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    return env


def train(bfs_tree):
    eps = 1.0
    steps = 0
    total_steps = 0
    best_eps_reward = 0.0
    d_eps = 1 - config.min_eps
    average10 = deque(maxlen=10)

    # get all stored edges
    tree_edges = bfs_tree.edges()

    for i in range(1, config.episodes + 1):
        state = env.reset()
        
        episode_steps = 0
        rewards = 0.0

        while True:
            action = agent.get_action(state['image'], epsilon=eps)
            steps += 1

            next_state, reward, done, _ = env.step(action[0])
            
            # randomly pop transitions from graph and remove it from tree
            batch_transitions = utils.sample_from_bfs(tree_edges=tree_edges, hash_table=table, batch_size=config.batch_size, device=device)

            loss, cql_loss, bellmann_error = agent.learn(batch_transitions)

            # TODO: update target network every x steps
            # update target network
            agent.soft_update(agent.network, agent.target_net)

            state = next_state
            rewards += reward
            episode_steps += 1

            if config.is_render:
                env.render()
            
            eps = max(1 - ((steps*d_eps)/config.eps_frames), config.min_eps)
            
            if done:
                break
        
        average10.append(rewards)
        total_steps += episode_steps
        
        print("Episode: {} | Reward: {} | Q Loss: {} | CQL Loss: {} | Bellman Error: {} | Steps: {} | Epsilon: {}".format(i, rewards, loss, cql_loss, bellmann_error, steps, eps))

        if rewards > best_eps_reward:
            best_eps_reward = rewards

            print("-> Best Model is Saved at Episode {} !".format(i))


if __name__ == "__main__":
    config = get_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # replay buffer graph
    graph = networkx.Graph()

    # load saved transitions
    replay_buffer = utils.open_dataset()

    # hash-table to represent states in hash integer
    table = HashTable(buffer_size=len(replay_buffer))

    # create environmenty and assign seeds
    env = make_env()

    agent = CQLAgent(state_size=env.observation_space['image'].shape, action_size=env.action_space.n, device=device)

    # create graph with hash-table
    graph, bfs_tree, table = build_graph(graph=graph, buffer_data=replay_buffer, table=table)

    # save deep copy of the buffer for regenerating memory
    table.save_buffer()
    
    # initialize training
    train(bfs_tree)
