import networkx
import utils
import numpy as np
import matplotlib.pyplot as plt
from hash_table import HashTable
from utils import sample_from_bfs


def build_graph(graph, buffer_data, table):
    # breadth-first-search trees
    trees = []

    # loop through each transition and store in the graph
    for idx, transition in enumerate(buffer_data):

        state = transition['state'].flatten()
        next_state = transition['next_state'].flatten()
        reward = transition['reward']
        done = transition['done']

        # store the terminal state so that from there, start bfs algorithm
        if done:
            terminal_state = state

            # apply reverse breadth-first-search and create an oriented tree
            bfs = networkx.bfs_tree(G=graph, source=hash(tuple(terminal_state)), reverse=True)
            trees.append(bfs)

        # concatenate states in one transition; used to differentiate different edges NOTE: check usage
        current_next = np.concatenate((state, next_state))

        # store transition inside the table
        table[tuple(state)] = transition

        # graph.add_edge(idx, idx + 1)#, weight=reward)
        graph.add_edge(hash(tuple(state)), hash(tuple(next_state)))

    return graph, trees


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
    graph, bfs_trees = build_graph(graph=graph, buffer_data=replay_buffer, table=table)

    plot_graph(G=graph)

    # get all stored edges
    tree_edges = bfs_trees[0].edges()

    # randomly pop transitions from graph and remove it from tree
    tree_edges, batch_transitions = sample_from_bfs(tree_edges=tree_edges, hash_table=table, batch_size=4, device='cpu')

    print("batch_transitions : ", batch_transitions)
