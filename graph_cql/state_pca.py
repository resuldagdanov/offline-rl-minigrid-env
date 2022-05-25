from platform import node
import torch
import utils
import numpy as np
import random
import argparse
import networkx
import os
import sys
import gym
import gym_minigrid
import matplotlib.pyplot as plt
import torch.nn.functional as F
from hash_table import HashTable
from construct_graph import plot_graph
from sklearn.decomposition import PCA

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
    parser.add_argument("--eps_frames", type=int, default=1e-5, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e-5")
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


def compute_td_error(current_network, target_network, transition):
    gamma = 0.99
    states, actions, rewards, next_states, dones = transition

    states = torch.from_numpy(np.reshape(states, (1, 7, 7, 3))).float().to(device)
    actions = torch.from_numpy(np.reshape(actions, (1, 1))).long().to(device)
    rewards = torch.from_numpy(np.reshape(rewards, (1, 1))).float().to(device)
    next_states = torch.from_numpy(np.reshape(next_states, (1, 7, 7, 3))).float().to(device)
    dones = torch.from_numpy(np.reshape(dones, (1, 1))).long().to(device)

    with torch.no_grad():
        Q_targets_next = target_network(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    
    Q_a_s = current_network(states)
    Q_expected = Q_a_s.gather(1, actions)
    
    td_error = F.mse_loss(Q_expected, Q_targets)

    return td_error.detach().item()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load saved transitions
    replay_buffer = utils.open_dataset()

    config = get_config()

    # replay buffer graph
    graph = networkx.Graph()

    # hash-table to represent states in hash integer
    table = HashTable(buffer_size=len(replay_buffer))
    
    # create environmenty and assign seeds
    env = make_env()

    agent = CQLAgent(state_size=env.observation_space['image'].shape, action_size=env.action_space.n, device=device)

    transitions = []
    all_states = []
    all_rewards = []
    all_dones = []
    errors = []
    bfs_trees = []

    # loop through each transition that is previously stored
    for idx, transition in enumerate(replay_buffer):

        state = transition['state'].flatten()
        next_state = transition['next_state'].flatten()
        reward = transition['reward']
        done = transition['done']

        trans = (transition['state'], transition['action'], transition['reward'], transition['next_state'], transition['done'])
        transitions.append(trans)

        td_error = compute_td_error(current_network=agent.network, target_network=agent.target_net, transition=trans)

        all_states.append(state)
        all_rewards.append(reward)
        all_dones.append(done)
        errors.append(td_error)

        # store the terminal state so that from there, start bfs algorithm
        if done:
            terminal_state = state

            # apply reverse breadth-first-search and create an oriented tree
            bfs = networkx.bfs_tree(G=graph, source=hash(tuple(terminal_state)), reverse=True)
            bfs_trees.append(bfs)

        # concatenate states in one transition; used to differentiate different edges NOTE: check usage
        current_next = np.concatenate((state, next_state))

        # store transition inside the table
        table[tuple(state)] = transition

        # graph.add_edge(idx, idx + 1)#, weight=reward)
        graph.add_edge(hash(tuple(state)), hash(tuple(next_state)))

    all_states = np.array(all_states)
    all_rewards = np.reshape(all_rewards, (len(all_rewards), 1))
    all_dones = np.reshape(all_dones, (len(all_dones), 1))
    errors = np.reshape(errors, (len(errors), 1))
    
    pca = PCA(n_components=1)

    pca_data = pca.fit_transform(all_states)

    stacked = np.hstack((pca_data, all_rewards, errors))

    pca = PCA(n_components=2)

    pca_stacked_data = pca.fit_transform(stacked)

    print("graph : ", graph)
    print("bfs_trees : ", bfs_trees, len(bfs_trees))

    print("len(pca_stacked_data) : ", len(pca_stacked_data))

    plot_graph(graph)

    pca_graph = networkx.Graph()
    pca_table = HashTable(buffer_size=len(pca_stacked_data))

    for idx in range(len(pca_stacked_data)):
        state, action, reward, next_state, done = transitions[idx]

        if idx < len(pca_stacked_data) - 1:
            in_node = pca_stacked_data[idx]
            out_node = pca_stacked_data[idx + 1]

            distance = np.linalg.norm(in_node - out_node)
            weight = 1 / (distance + 1e-5)

            # pca_graph.add_edge(hash(tuple(pca_stacked_data[idx])), hash(tuple(pca_stacked_data[idx + 1])))
            pca_graph.add_edges_from([tuple(in_node), tuple(out_node)], weight=weight)

            node_info = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'in_node': in_node,
                'out_node': out_node,
                'distance': distance,
                'weight': weight
                }
            
            pca_table[tuple(in_node)] = node_info
        
        source_node = hash(tuple(state.flatten()))
        target_node = hash(tuple(next_state.flatten()))

        if len(graph.get_edge_data(source_node, target_node)) == 0 or graph[source_node][target_node]['weight'] > weight:
            graph[source_node][target_node]['weight'] = weight
    
    print("graph : ", graph)

    plot_graph(graph)

    short_path = networkx.dijkstra_path(G=graph, source=hash(tuple(terminal_state)), target=None, weight='weight')

    print("short_path : ", short_path)

    nodes_list = list(short_path.values())

    print("nodes_list : ", nodes_list, type(nodes_list))

    one_sample_node = nodes_list[0][0]

    print("one_sample_node : ", one_sample_node)

    flatten_state, transition = table.get_with_key(hash_key=one_sample_node)

    print("pca_graph : ", pca_graph)

    print("! transition : ", transition)

    plot_graph(pca_graph)
        
    plt.scatter(pca_stacked_data[:, 0], pca_stacked_data[:, 1], alpha=0.5)
    plt.show()
