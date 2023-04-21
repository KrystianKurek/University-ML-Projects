import numpy as np
import networkx as nx

def create_diffusion_graph(G, start_node, number_of_nodes):
    '''
    Cretes a diffusion graph.

    Arguments:
        * G - nx.Graph,
        * start_node - node from which diffusion graph creation starts
        * number_of_nodes - number of vertices in a diffusion graph

    Returns:
        * G_diff - diffusion graph
    '''
    G_diff_vertices = [start_node]
    G_diff_edges = []

    for _ in range(number_of_nodes-1):
        
        while True:
            u = np.random.choice(G_diff_vertices)
            neighs = list(G.neighbors(u))
            not_taken_neighs = np.setdiff1d(neighs, G_diff_vertices)
            if len(not_taken_neighs)>0:
                break
    
        w = np.random.choice(not_taken_neighs)

        G_diff_vertices.append(w)
        G_diff_edges.append((u, w))

    G_diff = nx.MultiGraph()
    G_diff.add_nodes_from(G_diff_vertices)
    G_diff.add_edges_from(G_diff_edges)
    G_diff.add_edges_from(G_diff_edges)

    return G_diff

