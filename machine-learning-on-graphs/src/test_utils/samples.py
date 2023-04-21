import numpy as np

def get_random_graph_matrix(N):
    '''
    Generates a random adjacency matrix of size NxN

    Arguments:
        * N - matrix size

    Returns:
        * symmmetrics adjacency matrix with zeros on diagonal 
    '''
    b = np.random.randint(0,2,size=(N,N))
    b_symm = np.tril(b) + np.tril(b, -1).T
    for i in range(b_symm.shape[0]):
        b_symm[i, i] = 0
    return b_symm