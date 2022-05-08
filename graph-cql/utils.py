import argparse
import numpy as np
import random
import torch
import gym
import pickle
import copy


def create_config():
    parser = argparse.ArgumentParser(description='Offline-RL')

    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to be collected, default: 200")
    parser.add_argument("--buffer_size", type=int, default=1_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--eps_frames", type=int, default=1e3, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e-5")
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


def make_hash(o):

  """
  Makes a hash from a dictionary, list, tuple or set to any level, that contains
  only other hashable types (including any lists, tuples, sets, and
  dictionaries).
  """

  if isinstance(o, (set, tuple, list)):
    return tuple([make_hash(e) for e in o])    

  elif not isinstance(o, dict):
    return hash(o)

  new_o = copy.deepcopy(o)
  
  for k, v in new_o.items():
    new_o[k] = make_hash(v)

  return hash(tuple(frozenset(sorted(new_o.items()))))
