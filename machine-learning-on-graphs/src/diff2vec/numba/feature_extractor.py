import numpy as np
from numba import njit
from diff2vec.numba.diffuser import create_diffusion_graph_numba

@njit
def get_feature_vectors_numba(A, sliding_window_size, diff_graph_nodes_number):
    '''
    Generates feature vectors for each vertex in a given graph, based on its neighboring nodes and their positional relationships within a sliding window.

    Arguments:
    * A - adjacency matrix of the graph,
    * sliding_window_size - the size of the sliding window used to generate the feature vectors,
    * diff_graph_nodes_number - the number of nodes to be used in the diffusion graph.
    
    Returns:
    * v_feature_vectors - a 2D numpy array representing the feature vectors for each vertex in the graph, where each row represents a vertex
    '''
    sequences = get_graph_sequences_numba(A, diff_graph_nodes_number)

    windows_positions = np.array([pos for pos in range(-sliding_window_size, sliding_window_size+1) if pos!=0])

    v_feature_vectors = np.zeros((A.shape[1], windows_positions.shape[0]*A.shape[1])).astype("int")
    for v in range(A.shape[1]):
        v_feature_vectors[v] = get_feature_vector_for_vertex_numba(sequences, windows_positions, v)
    
    return v_feature_vectors, sequences


@njit
def get_feature_vector_for_vertex_numba(sequences, windows_positions, v):
    '''
    Generates a feature vector for a given vertex in a graph, based on its neighboring nodes and their positional relationships within a sliding window.

    Arguments:
    * sequences - a 2D numpy array representing the diffusion graph sequences of the graph,
    * windows_positions - a 1D numpy array representing the positions of the sliding window to be used for feature extraction,
    * v - the index of the vertex for which the feature vector is to be generated.

    Returns:
    * v_feature_vector - a 1D numpy array representing the feature vector for the given vertex, where each element corresponds to the count 
    of a neighboring node within the sliding window.
    '''
    num_vertices_in_graph = sequences.shape[0]
    v_feature_vector = np.zeros(len(windows_positions)*num_vertices_in_graph).astype("int")

    for i, window_pos in enumerate(windows_positions):
        window_pos_feature_vec = np.zeros(num_vertices_in_graph).astype("int")

        for sequence in sequences:
             v_idxs = np.where(sequence == v)[0]
             window_pos_idxs = v_idxs + window_pos
             window_pos_idxs = window_pos_idxs[(window_pos_idxs>=0) & (window_pos_idxs<len(sequence))]
             window_pos_values = sequence[window_pos_idxs]
             window_pos_feature_vec[window_pos_values] +=1

        v_feature_vector[i*num_vertices_in_graph:(i+1)*num_vertices_in_graph] = window_pos_feature_vec
    
    return v_feature_vector

@njit
def get_graph_sequences_numba(A, diff_graph_nodes_number):
    '''
    Generates a set of sequences for each vertex in a given graph, where each sequence represents 
    an Eulerian path through a diffusion graph starting at that vertex.

    Arguments:
    * A - adjacency matrix of the graph,
    * diff_graph_nodes_number - the number of nodes to be used in the diffusion graph.
    
    Returns:
    * G_euler_paths - a 2D numpy array representing the sequences for each vertex in the graph, where each row represents 
    a vertex and each column represents a node in the corresponding Eulerian path.
    '''
    G_euler_paths = np.zeros((A.shape[1], (diff_graph_nodes_number-1)*2)).astype("int")

    for v in range(A.shape[1]):
        diff_vertices, diff_edges = create_diffusion_graph_numba(A, v, diff_graph_nodes_number)
        euler_path = get_euler_path_from_eulerian_numba(diff_vertices, diff_edges)
        G_euler_paths[v, :] = euler_path

    return G_euler_paths

# gets the euler path by first transfomring graph to eulerian graph and then finding the euler path
@njit
def get_euler_path_from_eulerian_numba(vertices, edges):
    '''
    Generates the Euler path of a given graph represented by its vertices and edges, by first making it eulerian (doubling each edge).

    Arguments:
    * vertices - a 1D numpy array representing the vertices of the graph.
    * edges - a 2D numpy array representing the edges of the graph.

    Returns:
    * euler_path - a list of vertices representing the Euler path of the graph.
    '''
    # map vertices to subsequent numbers
    new_verices_labels = np.arange(vertices.shape[0])
    mapping = dict()
    revert_mapping = dict()
    for i in range(len(vertices)):
        mapping[vertices[i]] = new_verices_labels[i]
        revert_mapping[new_verices_labels[i]] = vertices[i]

    # create multigraph that is eulerian
    multi_A = np.zeros((vertices.shape[0], vertices.shape[0])).astype("int")
    for edge in edges:
        multi_A[mapping[edge[0]]][mapping[edge[1]]] = multi_A[mapping[edge[1]]][mapping[edge[0]]] = 2

    # get euler path
    euler_path_mapped = get_euler_path_iter_numba(multi_A)

    # revert mapping
    euler_path = [revert_mapping[v] for v in euler_path_mapped]

    return euler_path


# iterative algorithm
@njit
def get_euler_path_iter_numba(A):
    '''
    Description: This function takes an adjacency matrix of a graph and returns an Euler path of the graph, using an iterative algorithm.

    Arguments:
    * A: A 2D numpy array representing the adjacency matrix of the graph.

    Returns:
    * path: A 1D numpy array representing the Euler path of the graph.
    '''
    edges_count = np.where(A>0)[0].shape[0]
    stack = np.zeros(edges_count+1).astype("int")
    stack_idx = 0
    u = 0
    stack[stack_idx] = u

    path = np.zeros(edges_count+1).astype("int")
    path_idx = 0

    while stack_idx >= 0:
        v = stack[stack_idx] 

        found = False
        for i in range(A.shape[0]):
            if A[v,i] > 0:
                A[v,i]-=1
                A[i,v]-=1
                stack_idx += 1
                stack[stack_idx] = i
                found = True
                break
        if not found:
            stack_idx -= 1
            path[path_idx] = v
            path_idx+=1
    
    return path[:-1]



@njit
def dfs_euler_rec_numba(A, v, stack, stack_idx):
    '''
    Recursive DFS needed for get_euler_path_rec_numba.
    '''

    for i in range(A.shape[0]):
        while A[v,i] > 0:
            A[v,i]-=1
            A[i,v]-=1
            dfs_euler_rec_numba(A, i, stack, stack_idx)
            print(stack_idx)
    stack[stack_idx[0]] = v
    stack_idx[0]+=1

@njit
def get_euler_path_rec_numba(A):
    '''
    Description: This function takes an adjacency matrix of a graph and returns an Euler path of the graph, using an recursive algorithm.
    NOTE: recursion, not feasible for larger graphs (N>100) due to memeory limits!

    Arguments:
    * A: A 2D numpy array representing the adjacency matrix of the graph.

    Returns:
    * stack: A 1D numpy array representing the Euler path of the graph.
    '''
    edges_count = np.where(A>0)[0].shape[0]
    stack = np.zeros(edges_count+1).astype("int")
    stack_idx = np.array([0])
    
    dfs_euler_rec_numba(A, 0, stack, stack_idx)
    return stack[:-1]

