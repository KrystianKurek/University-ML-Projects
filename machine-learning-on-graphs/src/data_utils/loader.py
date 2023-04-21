import networkx as nx
from sklearn.model_selection import train_test_split
import numpy as np
import requests
import os

from tqdm.contrib import itertools


############################## train test split ##############################
def train_test_G_split(*args, seed=2137):
    '''
    Returns train test split with default sklearn parameters and radnom seed=2137

    Arguments:
        * args - argumetns to split

    Returns: 
        * train test splitted args

    '''
    seed = 2137
    return train_test_split(*args, random_state=seed)


def train_val_test_split(*args, seed=2137):
    first_split = train_test_split(*args, test_size=0.2, random_state=seed)
    train_val = [elem for i, elem in enumerate(first_split) if i % 2 == 0]
    test = [elem for i, elem in enumerate(first_split) if i % 2 == 1]
    second_split = train_test_split(*train_val, test_size=0.125, random_state=seed)
    train = [elem for i, elem in enumerate(second_split) if i % 2 == 0]
    val = [elem for i, elem in enumerate(second_split) if i % 2 == 1]
    result = []
    for (train_samples, val_samples, test_samples) in zip(train, val, test):
        result.append(train_samples)
        result.append(val_samples)
        result.append(test_samples)

    return result


############################## graph getters ##############################
def get_twitch_graph():
    '''
    Returns:
        * tuple (networkx.Graph, target labels) from twitch data (taken from karateclub repo)
    '''
    root_dir = "../data"
    data_dir = "twitch"
    edges_url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/de4cdb473b9992395159a2fd6732d027773c1c04/dataset/node_level/twitch/edges.csv'
    target_url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/de4cdb473b9992395159a2fd6732d027773c1c04/dataset/node_level/twitch/target.csv'

    return __get_graph(root_dir, data_dir, edges_url, target_url)


def get_lastfm_graph():
    '''
    Returns:
        * tuple (networkx.Graph, target labels) from lastfm data (taken from karateclub repo)
    '''
    root_dir = "../data"
    data_dir = "lastfm"
    edges_url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level/lastfm/edges.csv'
    target_url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level/lastfm/target.csv'

    return __get_graph(root_dir, data_dir, edges_url, target_url)


def get_communities_graph(n=1000, n_communities=4, p_low=0.07, p_high=0.8):
    comm_indices = np.arange(n).reshape(n_communities, -1)
    pairs = [np.array(list(itertools.product(comm_indices_, comm_indices_))) for comm_indices_ in comm_indices]
    am = np.random.choice([1, 0], p=[p_low, 1 - p_low], size=(n, n))
    am_high = np.random.choice([1, 0], p=[p_high, 1 - p_high], size=(n, n))

    for pairs_ in pairs:
        am[pairs_[:, 0], pairs_[:, 1]] = am_high[pairs_[:, 0], pairs_[:, 1]]

    am = np.triu(am, k=1)
    c_size = n // n_communities
    y = np.concatenate([i*np.ones(c_size) for i in range(n_communities)])
    return nx.from_numpy_array(am), y


############################## private functions ##############################
def __get_graph(root_dir, data_dir, edges_url, target_url):
    '''
    Downloads the graph and targets (edges_url, target_url) and saves (root_dir, data_dir)
    '''

    edges_path = os.path.join(root_dir, data_dir, "edges.csv")
    target_path = os.path.join(root_dir, data_dir, "target.csv")

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(os.path.join(root_dir, data_dir)):
        os.makedirs(os.path.join(root_dir, data_dir))

    __download_csv_data(edges_url, edges_path)
    __download_csv_data(target_url, target_path)

    return __get_graph_from_csv(edges_path, target_path)


def __get_graph_from_csv(edges_path, target_path):
    '''
    Reads the graph (edges_path) and targets (target_path) from csv file
    '''
    targets = np.loadtxt(target_path, skiprows=1, delimiter=',', dtype=int)[:,1]
    edges = np.loadtxt(edges_path, skiprows=1, delimiter=',', dtype=int)

    G = nx.Graph()
    G.add_nodes_from(range(len(targets)))
    G.add_edges_from(edges)

    return G, targets


def __download_csv_data(url, out_file_path):
    '''
    Downloads from url and saves to out_file_path
    '''
    res = requests.get(url, allow_redirects=True)
    with open(out_file_path, 'wb') as file:
        file.write(res.content)
