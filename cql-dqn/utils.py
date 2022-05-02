import os
import torch


def save(args, save_name, model, wandb, ep=None):
    save_dir = './trained_models/' 
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + "_" + save_name + "_eps" + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + "_" + save_name + "_eps" + str(ep) + ".pth")
    
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + "_" + save_name + ".pth")
        wandb.save(save_dir + args.run_name + "_" + save_name + ".pth")


def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    
    for _ in range(num_samples):

        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        
        dataset.add(state['image'], action, reward, next_state['image'], done)
        
        state = next_state
    
        if done:
            state = env.reset()


def collect_from_model(env, agent, dataset, num_samples=100):
    state = env.reset()
    
    for _ in range(num_samples):

        action = agent.get_greedy_action(state['image'])
        next_state, reward, done, _ = env.step(action)
        
        dataset.add(state['image'], action, reward, next_state['image'], done)
        
        state = next_state
    
        if done:
            state = env.reset()
