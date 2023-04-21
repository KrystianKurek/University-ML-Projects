import numpy as np
from numba import njit

@njit
def create_diffusion_graph_numba(A, start_node, number_of_nodes):
    '''
    Cretes a diffusion graph.

    Arguments:
        * A - graph adjacency matrix,
        * start_node - node from which diffusion graph creation starts
        * number_of_nodes - number of vertices in a diffusion graph

    Returns:
        * G_diff_vertices - np.array of diffusion graph vertices (|V|), 
        * G_diff_edges - np.array of edges in difussion graph (|E|x2)
    '''

    G_diff_vertices = np.zeros(number_of_nodes).astype("int")
    G_diff_vertices[0] = start_node
    
    free_vertices = np.ones(A.shape[1]).astype("int")
    free_vertices[start_node] = 0

    G_diff_edges = np.zeros((number_of_nodes-1, 2)).astype("int")

    for i in range(number_of_nodes-1):
        
        vertex_choice_trials = 0
        while True:
            u = np.random.choice(G_diff_vertices[0:i+1])
            neighs = A[u, :]
            not_taken_neighs = np.where(neighs * free_vertices)[0] 
            if len(not_taken_neighs)!=0:
                break
            elif (len(not_taken_neighs)==0) and (vertex_choice_trials>50):
                raise Exception("No vertices left in 50 trials")
            else:
                vertex_choice_trials+=1
        
        w = np.random.choice(not_taken_neighs)
        free_vertices[w]=0
        
        G_diff_vertices[i+1] = w
        G_diff_edges[i, :] = [u, w]

    return G_diff_vertices, G_diff_edges