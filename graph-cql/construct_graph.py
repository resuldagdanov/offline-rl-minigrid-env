import networkx
import utils
import numpy as np
import matplotlib.pyplot as plt
from hash_table import HashTable


def build_graph(graph, buffer_data):

    # loop through each transition and store in the graph
    for idx, transition in enumerate(buffer_data):

        state = transition['state'].flatten()
        next_state = transition['next_state'].flatten()
        reward = transition['reward']
        done = transition['done']

        # store the terminal state so that from there, start bfs algorithm
        if done:
            last_state = state

        # print("\nstate : ", np.reshape(state.flatten(), (7, 7, 3)), state.shape)

        # concatenate states in one transition; used to differentiate different edges NOTE: check usage
        current_next = np.concatenate((state, next_state))

        # store transition inside the table
        table[tuple(state)] = transition

        # graph.add_edge(idx, idx + 1)#, weight=reward)
        graph.add_edge(hash(tuple(state)), hash(tuple(next_state)))
    
    # apply reverse breadth-first-search and create an oriented tree
    bfs = networkx.bfs_tree(G=graph, source=hash(tuple(last_state)), reverse=True)

    return graph, bfs


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
    replay_buffer = utils.open_dataset()

    # hash-table to represent states in hash integer
    table = HashTable(buffer_size=len(replay_buffer))

    # create graph with hash-table
    graph, bfs = build_graph(graph=graph, buffer_data=replay_buffer)

    print("graph: ", len(graph.edges()))

    print("bfs : ", len(bfs.edges()))

    plot_graph(G=bfs)
