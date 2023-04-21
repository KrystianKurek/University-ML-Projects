import numpy as np
import networkx as nx
import tqdm
from diff2vec.diffuser import create_diffusion_graph


def get_feature_vectors(G, sliding_window_size, diff_graph_nodes_number, return_sequences=False):
    '''
    Generates feature vectors for each vertex in a given graph, based on its neighboring nodes and their positional relationships within a sliding window.

    Arguments:
    * G - nx.Graph,
    * sliding_window_size - the size of the sliding window used to generate the feature vectors,
    * diff_graph_nodes_number - the number of nodes to be used in the diffusion graph.
    * return_sequences - whether to return also graph sequences generated from difussion graph.
    
    Returns:
    * v_feature_vectors - a 2D numpy array representing the feature vectors for each vertex in the graph, where each row represents a vertex OR
    * v_feature_vectors, sequences - if return_sequences = True
    '''
    sequences = get_graph_sequences(G, diff_graph_nodes_number)

    windows_positions = [pos for pos in range(-sliding_window_size, sliding_window_size+1) if pos!=0]

    v_feature_vectors = np.zeros((len(G), len(windows_positions)*len(G)))
    for v_idx, v in enumerate(tqdm.tqdm(G.nodes)):
        v_feature_vectors[v_idx] = get_feature_vector_for_vertex(sequences, windows_positions, v, len(G))
    
    if return_sequences:
        return v_feature_vectors, sequences
    else:
        return v_feature_vectors



def get_feature_vector_for_vertex(sequences, windows_positions, v, num_vertices):
    '''
    Generates a feature vector for a given vertex in a graph, based on its neighboring nodes and their positional relationships within a sliding window.

    Arguments:
    * sequences - a 2D numpy array representing the diffusion graph sequences of the graph,
    * windows_positions - a 1D numpy array representing the positions of the sliding window to be used for feature extraction,
    * v - the index of the vertex for which the feature vector is to be generated.
    * num_vertices -  number of vertices in the entire graph

    Returns:
    * v_feature_vector - a 1D numpy array representing the feature vector for the given vertex, where each element corresponds to the count 
    of a neighboring node within the sliding window.
    '''
    v_feature_vector = np.zeros(len(windows_positions)*num_vertices)

    for i, window_pos in enumerate(windows_positions):
        window_pos_feature_vec = np.zeros(num_vertices)

        for sequence in sequences:
            sequence = np.array(sequence)
            v_idxs = np.where(sequence == v)[0]
            window_pos_idxs = v_idxs + window_pos
            window_pos_idxs = window_pos_idxs[(window_pos_idxs>=0) & (window_pos_idxs<len(sequence))]
            window_pos_values = sequence[window_pos_idxs]
            window_pos_feature_vec[window_pos_values] +=1

        v_feature_vector[i*num_vertices:(i+1)*num_vertices] = window_pos_feature_vec
    
    return v_feature_vector



def get_graph_sequences(G, diff_graph_nodes_number):
    '''
    Generates a set of sequences for each vertex in a given graph, where each sequence represents 
    an Eulerian path through a diffusion graph starting at that vertex.

    Arguments:
    * G - nx.Graph,
    * diff_graph_nodes_number - the number of nodes to be used in the diffusion graph.
    
    Returns:
    * G_euler_paths - a 2D numpy array representing the sequences for each vertex in the graph, where each row represents 
    a vertex and each column represents a node in the corresponding Eulerian path.
    '''
    G_euler_paths = []

    for v in G.nodes:
        G_diff = create_diffusion_graph(G, v, diff_graph_nodes_number)
        euler_path = get_euler_path(G_diff)
        G_euler_paths.append(euler_path)

    return G_euler_paths



def get_euler_path(G):
    '''
    Arguments:
        * G - nx.Graph

    Returns:
        * euler path in G
    '''
    return [x[0] for x in list(nx.eulerian_path(G))]





