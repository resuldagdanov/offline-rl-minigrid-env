import networkx
import utils
import numpy as np
import matplotlib.pyplot as plt
from hash_table import HashTable


def plot_graph(G):
    subax1 = plt.subplot(121)
    networkx.draw(G, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    networkx.draw_shell(G, with_labels=True, font_weight='bold')
    plt.show()


if __name__ == "__main__":

    # replay buffer graph
    graph = networkx.Graph()

    # load saved transitions
    buffer_data = utils.open_dataset()

    buffer_size = len(buffer_data)
    # hash-table to represent states in hash integer
    table = HashTable(buffer_size=buffer_size)

    # loop through each transition and store in the graph
    for idx, transition in enumerate(buffer_data):

        state = transition['state'].flatten()
        next_state = transition['next_state'].flatten()
        reward = transition['reward']
        done = transition['done']

        # print("\nstate : ", np.reshape(state.flatten(), (7, 7, 3)), state.shape)

        # concatenate states in one transition; used to differentiate different edges
        current_next = np.concatenate((state, next_state))

        # store transition inside the table
        table[tuple(current_next)] = transition

        print("\n")
        print(table[tuple(current_next)])

        # graph.add_edge(idx, idx + 1)#, weight=reward)
        #graph.add_edge(hash(tuple(state)), hash(tuple(next_state)))

    print("graph: ", graph)

    plot_graph(G=graph)
