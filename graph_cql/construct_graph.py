from sys import get_coroutine_origin_tracking_depth
import networkx
import numpy as np
import matplotlib.pyplot as plt
from hash_table import HashTable
from utils import open_dataset, state2hash
from collections import namedtuple


def build_graph(graph, table, buffer_data):
    # loop through each transition and store in the graph
    for transition in buffer_data:

        state = transition['state'].flatten()
        next_state = transition['next_state'].flatten()
        action = float(transition['action'])
        reward = float(transition['reward'])
        done = int(transition['done'])

        # get unique hash id for the state index in the buffer
        current_state_hash = state2hash(state)
        next_state_hash = state2hash(next_state)

        # store transition inside the table
        table[current_state_hash] = transition['state']
        table[next_state_hash] = transition['next_state']

        # create an edge from the current state and the next state
        # add reward and termination state atributes to the edge
        graph.add_edge(current_state_hash, next_state_hash, reward=reward, action=action, done=done, weight=1)

    return graph, table


def plot_graph(G):

    subax1 = plt.subplot(121)
    networkx.draw(G, with_labels=False, font_weight='bold')
    
    subax2 = plt.subplot(122)
    networkx.draw_shell(G, with_labels=False, font_weight='bold')
    
    plt.show()


if __name__ == "__main__":

    # replay buffer graph
    graph = networkx.Graph()

    # load saved transitions
    replay_buffer = open_dataset()

    # hash-table to represent states in hash integer
    table = HashTable(buffer_size=len(replay_buffer))

    # create graph with hash-table
    graph, table = build_graph(graph=graph, buffer_data=replay_buffer, table=table)

    plot_graph(G=graph)

    # get all stored edges
    tree_edges = graph.edges()
