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
import torch.nn.functional as F
from collections import deque
from construct_graph import build_graph
from hash_table import HashTable
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


def states_2_pca(dataset, n_components):
    all_flat_states = []
    all_rewards = []

    # loop through each transition that is previously stored
    for transition in dataset:
        state = transition['state'].flatten()
        reward = transition['reward']

        all_flat_states.append(state)
        all_rewards.append(reward)

    all_flat_states = np.array(all_flat_states)

    state_pca = PCA(n_components=n_components)
    pca_data = state_pca.fit_transform(all_flat_states)

    return pca_data, np.reshape(all_rewards, (len(all_rewards), 1))


def pca_stacked_features(futures_matrix, n_components):
    feature_pca = PCA(n_components=n_components)
    embeddings = feature_pca.fit_transform(futures_matrix)

    return embeddings


def compute_td_error(current_network, target_network, transition):
    gamma = 0.99
    states, actions, rewards, next_states, dones = transition['state'], transition['action'], transition['reward'], transition['next_state'], transition['done']

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


def get_td_errors(current_network, target_network, dataset):
    errors = []

    for transition in dataset:
        td_error = compute_td_error(current_network, target_network, transition)
        errors.append(td_error)

    return np.reshape(errors, (len(errors), 1))


def update_edge_weights(graph, embeddings, dataset):

    for idx in range(len(embeddings) - 1):
        in_node = embeddings[idx]
        out_node = embeddings[idx + 1]

        distance = np.linalg.norm(in_node - out_node)
        weight = 1 / (distance + 1e-5)

        source_node = hash(tuple(dataset[idx]['state'].flatten()))
        target_node = hash(tuple(dataset[idx]['next_state'].flatten()))

        # graph.add_edges_from([source_node, target_node], weight=weight)
        if len(graph.get_edge_data(source_node, target_node)) == 0 or graph[source_node][target_node]['weight'] > weight:
            graph[source_node][target_node]['weight'] = weight


def train():
    eps = 1.0
    steps = 0
    k_tree = 0
    total_steps = 0
    best_eps_reward = 0.0
    d_eps = 1 - config.min_eps
    average10 = deque(maxlen=10)

    # get all stored edges
    tree_edges = bfs_trees[k_tree].edges()

    for i in range(1, config.episodes + 1):
        state = env.reset()
        
        episode_steps = 0
        rewards = 0.0

        while True:

            if len(tree_edges) < config.batch_size:
                # print("Tree edges:", k_tree, len(tree_edges))
                k_tree += 1

                # when maximum number of trees is reached, break the loop
                if k_tree == len(bfs_trees):
                    # print("Maximum number of trees reached !")
                    # return
                    k_tree = 0

                # get all stored edges
                tree_edges = bfs_trees[k_tree].edges()
                continue
            
            action = agent.get_action(state['image'], epsilon=eps)
            steps += 1

            next_state, reward, done, _ = env.step(action[0])
            
            # randomly pop transitions from graph and remove it from tree
            tree_edges, batch_transitions = utils.sample_from_bfs(tree_edges=tree_edges, hash_table=table, batch_size=config.batch_size, device=device)

            loss, cql_loss, bellmann_error = agent.learn(batch_transitions)

            IS_PCA = False
            if IS_PCA:

                # compute current TD error for all transitions in the buffer hash table
                error_component = get_td_errors(current_network=agent.network, target_network=agent.target_net, dataset=replay_buffer)

                # concatenate state, reward, and td-error features
                stacked_features = np.hstack((states_component, reward_samples, error_component))

                print("stacked_features : ", stacked_features, len(stacked_features))

                # take pca of the stacked features
                embeddings = pca_stacked_features(futures_matrix=stacked_features, n_components=2)

                print("embeddings : ", embeddings, len(embeddings))

                # compute distance of transitions in embedded space and update weights of edges in graph
                update_edge_weights(graph=graph, embeddings=embeddings, dataset=replay_buffer)

                print("weights are updated")

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
        
        print("Episode: {} | Reward: {} | Q Loss: {} | Steps: {}".format(i, rewards, loss, steps))

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

    # get pca component 1 from states
    states_component, reward_samples = states_2_pca(dataset=replay_buffer, n_components=1)

    # hash-table to represent states in hash integer
    table = HashTable(buffer_size=len(replay_buffer))

    # create environmenty and assign seeds
    env = make_env()

    agent = CQLAgent(state_size=env.observation_space['image'].shape, action_size=env.action_space.n, device=device)

    # create graph with hash-table
    graph, bfs_trees = build_graph(graph=graph, buffer_data=replay_buffer, table=table)

    print("graph : ", graph)
    print("bfs_trees : ", bfs_trees, len(bfs_trees))

    train()
