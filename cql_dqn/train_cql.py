import gym
import gym_minigrid
import random
import pybullet_envs
import numpy as np
import torch
import wandb
import argparse
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from replay_buffer import ReplayBuffer
from cql_agent import CQLAgent
from utils import save, collect_random, collect_from_model


def get_config():
    parser = argparse.ArgumentParser(description='Offline-RL')

    parser.add_argument("--run_name", type=str, default="cql-dqn", help="Run name, default: CQL-DQN")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=600, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=10_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini batch size, default: 32")
    parser.add_argument("--num_samples", type=int, default=10_000, help="Number of samples to collect, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 0.01")
    parser.add_argument("--eps_frames", type=int, default=1e3, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e-5")
    parser.add_argument("--is_render", type=int, default=0, help="Render environment during training when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--model_path", type=str, default="./trained_models/cql-dqn_mini-grid_random-agent_eps300.pth", help="Directory of the loaded model")
    parser.add_argument("--is_collect_from_model", type=int, default=0, help="Collect dataset from pre-trained agent model when set to 1, default: 0")
    
    args = parser.parse_args()
    return args


def make_env(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    env = gym.make(config.env)
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    return env


def train(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create environmenty and assign seeds
    env = make_env(config)

    eps = 1.0
    d_eps = 1 - config.min_eps
    steps = 0
    total_steps = 0
    average10 = deque(maxlen=10)
    
    with wandb.init(project="CQL", name=config.run_name, config=config):

        agent = CQLAgent(state_size=env.observation_space['image'].shape, action_size=env.action_space.n, device=device)

        wandb.watch(agent.network, log="gradients", log_freq=10)

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)

        # collect data by moving trained model
        if config.is_collect_from_model:
            agent.network.load_state_dict(torch.load(config.model_path))
            agent.network.eval()

            collect_from_model(env=env, agent=agent, dataset=buffer, num_samples=config.num_samples)
            model_save_name = "mini-grid_trained-agent"
        
        # collect data by applying random actions
        else:
            collect_random(env=env, dataset=buffer, num_samples=config.num_samples)
            model_save_name = "mini-grid_random-agent"

        best_eps_reward = 0.0
        
        for i in range(1, config.episodes + 1):
            state = env.reset()
            
            episode_steps = 0
            rewards = 0.0

            while True:
                action = agent.get_action(state['image'], epsilon=eps)
                steps += 1
                
                next_state, reward, done, _ = env.step(action[0])

                # TODO: remove adding to buffer as CQL is static dataset training
                # buffer.add(state['image'], action, reward, next_state['image'], done)

                batch_transitions = buffer.sample()
                loss, cql_loss, bellmann_error = agent.learn(batch_transitions)

                # ------------------- update target network ------------------- #
                # TODO: update target network every x steps
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

            writer.add_scalar("Vanilla-CQL-episode-reward", rewards, i)
            writer.add_scalar("Vanilla-CQL-steps-reward", rewards, total_steps)
            
            print("Episode: {} | Reward: {} | Q Loss: {} | Steps: {} | Epsilon: {}".format(i, rewards, loss, steps, eps))
            
            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Q Loss": loss,
                       "CQL Loss": cql_loss,
                       "Bellmann error": bellmann_error,
                       "Steps": steps,
                       "Epsilon": eps,
                       "Episode": i,
                       "Buffer size": buffer.__len__()})

            if i % config.save_every == 0:
                save(config, save_name=model_save_name, model=agent.network, wandb=wandb, ep=str(i))
            
            if rewards > best_eps_reward:
                best_eps_reward = rewards

                save(config, save_name=model_save_name, model=agent.network, wandb=wandb, ep=str(i) + "_best")
                print("-> Best Model is Saved at Episode {} !".format(i))


if __name__ == "__main__":
    config = get_config()

    # initialize tensorboard logging directory
    writer = SummaryWriter(log_dir="runs/")

    train(config)
