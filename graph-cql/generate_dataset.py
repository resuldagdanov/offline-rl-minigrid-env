import gym_minigrid
import utils
from collections import deque, namedtuple


def run_agent(config):

    # create replay memory buffer
    buffer = deque(maxlen=config.buffer_size)
    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    # create environmenty and assign seeds
    env = utils.make_environment(config)

    # randomly run agent to collect transitions
    buffer = utils.collect_transitions(env=env, dataset=buffer, experience=experience, num_steps=config.num_steps)

    return buffer


if __name__ == "__main__":

    # create config
    config = utils.create_config()

    # create list of transition steps
    dataset = run_agent(config)

    # saving buffer dataset
    utils.save_dataset(dataset)
