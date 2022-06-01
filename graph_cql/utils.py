import argparse
import random
import torch
import numpy as np
import gym
import pickle
import networkx
import torch.nn.functional as F


def create_config():
    parser = argparse.ArgumentParser(description='Offline-RL')

    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--num_steps", type=int, default=10_000, help="Number of steps to be collected, default: 200")
    parser.add_argument("--buffer_size", type=int, default=10_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--is_render", type=int, default=0, help="Render environment during training when set to 1, default: 0")
    
    args = parser.parse_args()

    return args


def collect_transitions(env, dataset, experience, num_steps):
    state = env.reset()
    
    for _ in range(num_steps):

        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        exp = experience(state['image'], action, reward, next_state['image'], done)
        dataset.append(exp._asdict())

        state = next_state
    
        if done:
            state = env.reset()

    return dataset


def make_environment(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    env = gym.make(config.env)
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    return env


def save_dataset(data):
    with open('dataset.pkl', 'wb') as file:
        pickle.dump(data, file)


def open_dataset():
    with open('dataset.pkl', 'rb') as file:
        data = pickle.load(file)

    return data


def state2hash(state):

    return hash(tuple(state))


def construct_transition(graph, hash_table, edges):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    
    for edge in edges:
        current_state_hash = edge[0]
        next_state_hash = edge[1]

        # re-construct transition
        states.append(hash_table[current_state_hash])
        next_states.append(hash_table[next_state_hash])
        actions.append(graph[current_state_hash][next_state_hash]['action'])
        rewards.append(graph[current_state_hash][next_state_hash]['reward'])
        dones.append(graph[current_state_hash][next_state_hash]['done'])
    
    return (states, actions, rewards, next_states, dones)


def convert_to_torch(states, actions, rewards, next_states, dones, device):
    states = torch.from_numpy(np.array(states)).float().to(device)
    actions = torch.from_numpy(np.array(actions)).float().to(device)
    actions = actions.type(torch.int64).unsqueeze(1)
    rewards = torch.from_numpy(np.reshape(rewards, (len(rewards), 1))).float().to(device)
    next_states = torch.from_numpy(np.array(next_states)).float().to(device)
    dones = torch.from_numpy(np.reshape(dones, (len(dones), 1))).float().to(device)

    return states, actions, rewards, next_states, dones


def compute_td_errors(agent, transition):
    states, actions, rewards, next_states, dones = transition

    with torch.no_grad():
        Q_targets_next = agent.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (agent.gamma * Q_targets_next * (1 - dones))
    
    Q_a_s = agent.network(states)
    Q_expected = Q_a_s.gather(1, actions)
    
    # td_error = F.mse_loss(Q_expected, Q_targets)
    td_errors = abs(Q_expected - Q_targets) ** 2
    
    return td_errors.detach().cpu().numpy()


def update_edge_weights(graph, edges, hash_table, agent, device):
    # re-construct transition from the given edges
    transition = construct_transition(graph=graph, hash_table=hash_table, edges=edges)

    # compute td errors of all transitions
    td_errors = compute_td_errors(agent=agent, transition=convert_to_torch(*transition, device=device))

    # filter out negative td errors
    #td_errors[td_errors < 0] = 0.0

    for idx, edge in enumerate(edges):
        graph[edge[0]][edge[1]]['weight'] = 1 / (float(abs(td_errors[idx])) + 1e-5)


def sample_with_random_walk(graph, hash_table, batch_size, seed, device):
    # run betweenness centrality on the graph and retrun edges
    walker_edges = networkx.edge_betweenness_centrality(G=graph, normalized=True, weight='weight', seed=seed)
    # tree_edges = networkx.minimum_spanning_edges(G=graph, algorithm='kruskal', weight='weight', data=True)

    # list of random walked edges
    selected_edges = list(walker_edges.keys())# [batch_size:]
    # selected_edges = random.sample(selected_edges, batch_size)
    # selected_edges = list(tree_edges)[:batch_size]

    print("\n")
    for i, edge in enumerate(selected_edges):
        print(i, graph[edge[0]][edge[1]]['reward'], graph[edge[0]][edge[1]]['weight'])

    print("walker_edges values : ", walker_edges.values())

    # re-construct transition from the given popped edges
    transition = construct_transition(graph=graph, hash_table=hash_table, edges=selected_edges)
    
    return convert_to_torch(*transition, device=device)


def sample_with_tsp(graph, hash_table, batch_size, seed, device):

    tsp = networkx.approximation.traveling_salesman_problem

    path = tsp(graph, weight='weight')

    print("path : ", path, len(path))

    print(len([x for x in hash_table.buffer if x is not None]))

    w_1 = graph[path[-1]][path[-2]]['weight']
    w_2 = graph[path[1]][path[0]]['reward']

    print(w_1, w_2)
    
    edges = graph.edges(data=True)

    # list of random walked edges
    selected_edges = list(tree_edges)[:batch_size]

    # re-construct transition from the given popped edges
    transition = construct_transition(graph=graph, hash_table=hash_table, edges=selected_edges)
    
    return convert_to_torch(*transition, device=device)
