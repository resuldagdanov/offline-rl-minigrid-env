import argparse
import numpy as np
import random
import torch
import gym
import pickle


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


def sample_from_bfs(tree_edges, hash_table, batch_size, device):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    tree_size = len(tree_edges)

    # number of samples must be included in each batch of transitions
    n_fixed_index = 1

    if tree_size < (batch_size - n_fixed_index):
        is_replace = True
    else:
        is_replace = False

    # first samples are corresponding to transitions with reward due to breadth-first-search algorithm
    fixed_indices = np.random.choice(a=range(0, n_fixed_index), size=n_fixed_index, replace=is_replace)

    # randomly sample transitions from the tree list
    random_indices = np.random.choice(range(n_fixed_index, tree_size), size=(batch_size - n_fixed_index), replace=is_replace)

    selected_indices = np.concatenate((fixed_indices, random_indices))
    np.random.shuffle(selected_indices)

    # get edges (current state hash, next state hash) from the tree list
    poped_edges = np.take(a=tree_edges, indices=selected_indices, axis=0)

    # as each edge stores a value of transition, look up to hash-table
    for edge in poped_edges:

        current_state_hash = edge[0]
        next_state_hash = edge[1]

        # transition of this edge is stored within the current state hash
        transition = hash_table[current_state_hash]

        states.append(transition['state'])
        actions.append(transition['action'])
        rewards.append(transition['reward'])
        next_states.append(transition['next_state'])
        dones.append(transition['done'])
    
    # convert lists of batch samples to torch device tensor
    states = torch.from_numpy(np.array(states)).float().to(device)
    actions = torch.from_numpy(np.array(actions)).float().to(device)
    actions = actions.type(torch.int64).unsqueeze(1)
    rewards = torch.from_numpy(np.reshape(rewards, (len(rewards), 1))).float().to(device)
    next_states = torch.from_numpy(np.array(next_states)).float().to(device)
    dones = torch.from_numpy(np.reshape(dones, (len(dones), 1))).float().to(device)

    return (states, actions, rewards, next_states, dones)


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


def update_weights(graph, edges, hash_table, agent, device):
    # re-construct transition from the given edges
    transition = construct_transition(graph=graph, hash_table=hash_table, edges=edges)

    # compute td errors of all transitions
    td_errors = compute_td_errors(agent=agent, transition=convert_to_torch(*transition, device=device))

    for idx, edge in enumerate(edges):
        graph[edge[0]][edge[1]]['weight'] = 1 / (float(abs(td_errors[idx])) + 1e-5)


def weighted_sample(current_state_hash, hash_table, agent, batch_size, device):
    selected_transitions = []

    for idx in range(batch_size):
        transitions = hash_table[current_state_hash]

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition in transitions:
            states.append(transition['state'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            next_states.append(transition['next_state'])
            dones.append(transition['done'])

        transitions = convert_to_torch(states, actions, rewards, next_states, dones, device)

        # compute td errors of all transitions
        td_errors = compute_td_errors(agent=agent, transition=convert_to_torch(*transition, device=device))
        td_errors = td_errors.detach().cpu().numpy() + 1e-5

        probabilities = td_errors / np.sum(td_errors)

        selected = np.random.choice(transitions, size=1, p=probabilities, replace=False)
        selected_transitions.append(selected)

        current_state_hash = selected['state']

    np.random.shuffle(selected_transitions)

    states, actions, rewards, next_states, dones = [], [], [], [], []
    for trans in selected_transitions:

        states.append(trans['state'])
        actions.append(trans['action'])
        rewards.append(trans['reward'])
        next_states.append(trans['next_state'])
        dones.append(trans['done'])
    
    states = torch.from_numpy(np.array(states)).float().to(device)
    actions = torch.from_numpy(np.array(actions)).float().to(device)
    actions = actions.type(torch.int64).unsqueeze(1)
    rewards = torch.from_numpy(np.reshape(rewards, (len(rewards), 1))).float().to(device)
    next_states = torch.from_numpy(np.array(next_states)).float().to(device)
    dones = torch.from_numpy(np.reshape(dones, (len(dones), 1))).float().to(device)
    
    return (states, actions, rewards, next_states, dones)
