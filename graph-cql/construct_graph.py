import networkx
import utils
import matplotlib.pyplot as plt


def plot_graph(G):
    subax1 = plt.subplot(121)
    networkx.draw(G, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    networkx.draw_shell(G, with_labels=True, font_weight='bold')
    plt.show()


if __name__ == "__main__":

    graph = networkx.Graph()

    buffer_data = utils.open_dataset()

    for idx, transition in enumerate(buffer_data):

        state = transition['state']
        next_state = transition['next_state']
        reward = transition['reward']
        done = transition['done']

        graph.add_edge(idx, idx + 1)#, weight=reward)

    print("graph: ", graph)

    plot_graph(G=graph)
