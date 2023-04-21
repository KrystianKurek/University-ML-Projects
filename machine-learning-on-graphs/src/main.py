import os
from datetime import datetime

import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA

from diff2vec.numba.feature_extractor import get_feature_vectors_numba
import tensorflow as tf
import numpy as np
import argparse
from data_utils.loader import get_twitch_graph, train_val_test_split, get_lastfm_graph, get_communities_graph
from models.models import return_simple_autoencoder, return_autoencoder, return_callbacks
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

def str2bool(v):
    """
    Convert a string representation of a boolean value to its corresponding boolean value.

    Args:
        v (str): The string representation of the boolean value.

    Returns:
        bool: The boolean value represented by the input string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args():
    '''
    Parse command line arguments using the argparse module.

    Arguments:
    None

    Returns:
    * args - an object containing parsed command line arguments
    '''
    
    parser = argparse.ArgumentParser(description='Argument Parser for Model Training')

    # Required arguments
    parser.add_argument('dataset', type=str, choices=['twitch', 'lastfm', 'artificial'],
                        help='The dataset to use for training (either "twitch", "lastfm", or "artificial")')

    parser.add_argument('architecture', type=str, choices=['diff2vec', 'AE', 'AE_with_DLM'],
                        help='The architecture to use for the model (either "diff2vec", "AE", or "DLM")')
    # Optional arguments
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='The size of the latent dimension for the model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The size of the batch')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=4,
                        help='Number of epochs without improvement to stop training')
    parser.add_argument('--n_blocks', type=int, default=2,
                        help='The number of residual blocks to use in the model')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='The number of filters to use in the convolutional layers')
    parser.add_argument('--use_maxpooling', type=str2bool, default=True,
                        help='Whether to use max pooling in the model or not')
    parser.add_argument('--use_upsampling', type=str2bool, default=True,
                        help='Whether to use upsampling in the model or not')
    parser.add_argument('--sliding_window_size', type=int, default=10,
                        help='The size of the sliding window for the diff2vec model')
    parser.add_argument('--diff_graph_nodes_number', type=int, default=20,
                        help='The number of nodes in the diffusion graph for the diff2vec model')
    parser.add_argument('--best_model_filepath', type=str, default='../models',
                        help='The path to save the best model')
    parser.add_argument('--summary_path', type=str, default='../summary',
                        help='The path to save the training summary')
    parser.add_argument('--features_path', type=str, default='../data_features',
                        help='The path to save the training summary')
    parser.add_argument('--use_true_labels', type=str2bool, default=True,
                        help='Whether to use the true labels for training or not')

    args = parser.parse_args()

    return args


def return_dataset(args):
    '''
    Select the appropriate dataset based on the parsed command line argument.

    Arguments:
        * args - object containing parsed command line arguments

    Returns:
        * Graph - the selected dataset as a NetworkX graph object
    '''
    if args.dataset == 'twitch':
        return get_twitch_graph()
    elif args.dataset == 'lastfm':
        return get_lastfm_graph()
    elif args.dataset == 'artificial':
        return get_communities_graph()
    else:
        raise ValueError(f"Incorrect dataset: {args.dataset}")


def return_model(args, x_shape, all_classes, sequences, now):
    '''
    Parse command line arguments to select an appropriate model based on the chosen architecture.

    Arguments:
        * args - an object containing parsed command line arguments
        * x_shape - shape of the input data
        * all_classes - a list of all possible classes in the dataset
        * sequences - a list of sequences of tokens
        * now - current time

    Returns:
        * An appropriate model object based on the chosen architecture from the command line arguments:
            * If args.architecture is 'AE', return a simple autoencoder model.
            * If args.architecture is 'AE_with_DLM', return an autoencoder model with a dynamic label multiplier.
            * If args.architecture is 'diff2vec', return a Word2Vec model.
            * Raises a ValueError if args.architecture is not a valid option.
    '''
    if args.architecture == 'AE':
        return return_simple_autoencoder(args.latent_dim,
                                         x_shape,
                                         num_filters=args.num_filters,
                                         use_maxpooling=args.use_maxpooling,
                                         use_upsampling=args.use_upsampling,
                                         n_blocks=args.n_blocks)
    elif args.architecture == 'AE_with_DLM':
        return return_autoencoder(args.latent_dim,
                                  x_shape,
                                  num_filters=args.num_filters,
                                  use_maxpooling=args.use_maxpooling,
                                  use_upsampling=args.use_upsampling,
                                  n_blocks=args.n_blocks,
                                  all_classes=all_classes,
                                  use_true_labels=args.use_true_labels,
                                  now=now,
                                  batch_size=args.batch_size,
                                  patience=args.patience,
                                  best_model_filepath=args.best_model_filepath)
    elif args.architecture == 'diff2vec':
        return Word2Vec(sequences,
                        vector_size=10,
                        window=10,
                        min_count=1,
                        sg=1,
                        workers=4,
                        alpha=0.025)
    else:
        raise ValueError(f"Incorrect architecture: {args.architecture}")


def return_features(G, args):
    '''
    This function generates and returns feature vectors and sequences for a given graph based on command line arguments. 
    If feature vectors and sequences already exist in the specified path, then it returns those instead of recomputing.

    Arguments:
        * G - the graph to compute features for
        * args - command line arguments containing dataset, sliding window size and diff graph nodes number

    Returns:
        * features - a 2D numpy array of shape (num_nodes, num_features) containing feature vectors for each node in the graph
        * sequences - a list of lists, where each sublist is a sequence of nodes in the graph used for the creation of the feature vectors

    '''
    dataset, sliding_window_size, diff_graph_nodes_number = args.dataset, args.sliding_window_size, \
        args.diff_graph_nodes_number
    os.makedirs(args.features_path, exist_ok=True)
    features_path = os.path.join(args.features_path, f"{dataset}_{sliding_window_size}_{diff_graph_nodes_number}_f.npy")
    sequences_path = os.path.join(args.features_path, f"{dataset}_{sliding_window_size}_{diff_graph_nodes_number}_s.npy")
    if not (os.path.exists(features_path) and os.path.exists(sequences_path)):
        features, sequences = get_feature_vectors_numba(nx.to_numpy_array(G).astype("int"), sliding_window_size,
                                                        diff_graph_nodes_number)
        np.save(features_path, features)
        np.save(sequences_path, sequences)
    else:
        features = np.load(features_path)
        sequences = np.load(sequences_path)
    sequences = [list(row) for row in sequences]
    return features, sequences


def reshape_array(X):
    '''
    Reshapes a given 4D numpy array to a new shape (n, m, p, X.shape[-1]), where m and p are the closest divisors
    of the product of dimensions of the array except the last one. This function is useful for reshaping image data to feed 
    to convolutional neural networks.

    Arguments:
    * X - a 4D numpy array of shape (n, m', p', c)

    Returns:
    * X_reshaped - a 4D numpy array of shape (n, m, p, c), where m and p are the closest divisors of m' * p'
    '''
    # Compute the new shape of the array
    n = X.shape[0]
    total_elements = np.prod(X.shape[1:-1])
    m, p = find_closest_divisors(total_elements)
    new_shape = (n, m, p, X.shape[-1])

    # Reshape the array
    X_reshaped = X.reshape(new_shape)

    return X_reshaped


def find_closest_divisors(n):
    '''
    finds the two divisors of a given number n that are the closest together.

    Arguments:
        * n - an integer number

    Returns:
        * Two integers i, j - two divisors of a given number n that are the closest together.
    '''
    # Find the two divisors of n that are the closest together
    for i in range(int(np.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i


def process_input(X, sliding_window_size):
    '''
    Reshapes and normalizes input array for further processing.

    Arguments:
    * `X` - input array with shape (n, d), where n is the number of nodes and d is the number of features
    * `sliding_window_size` - size of sliding window for feature extraction

    Returns:
    * `X_processed` - processed input array with shape (n, m, p, 1), where n is the number of nodes and m, p are the shape of the reshaped features after sliding window and reshaping
    '''
    X = X.reshape(len(X), 2 * sliding_window_size, -1)
    sum_ = X.sum(-1)
    sum_[sum_ == 0] = 1.0
    return reshape_array((X / sum_[:, :, np.newaxis])[:, :, :, np.newaxis])


def preprocess_features(*X, args):
    '''
    Preprocesses input feature arrays for further processing based on the specified architecture.

    Arguments:
    * `X` - variable length argument list of input arrays with shape (n, d), where n is the number of nodes and d is the number of features
    * `args` - command-line arguments parsed by argparse

    Returns:
    * `X_processed` - preprocessed input arrays based on the specified architecture:
    * If `args.architecture` is 'diff2vec', returns `X` unmodified.
    * If `args.architecture` is 'AE_with_DLM' or 'AE', applies sliding window and normalization using `process_input()` to each input array in `X`.
    * Raises a `ValueError` if `args.architecture` is not recognized.
    '''
    if args.architecture == 'diff2vec':
        return X
    elif args.architecture in {'AE_with_DLM', 'AE'}:
        return [process_input(x_set, args.sliding_window_size) for x_set in X]
    else:
        raise ValueError(f"Incorrect architecture: {args.architecture}")


def fit_model(args, model, X_train, X_val, y_train, y_val):
    '''
    Trains a machine learning model based on the specified architecture.

    Arguments:
    * args - command line arguments that specify the model configuration
    * model - the machine learning model to be trained
    * X_train - the input training data
    * X_val - the input validation data
    * y_train - the target training data
    * y_val - the target validation data
    
    Returns:
    * model - the trained machine learning model
    * history - the training history of the model if architecture is 'AE', otherwise None
    '''
    if args.architecture == 'AE':
        callbacks = return_callbacks(best_model_filepath=args.best_model_filepath, patience=args.patience)
        history = model.fit(X_train, X_train,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            shuffle=True,
                            validation_data=(X_val, X_val), callbacks=callbacks)
        model.load_weights(os.path.join(args.best_model_filepath, 'best_model.ckpt'))
        return model, history.history
    elif args.architecture == 'AE_with_DLM':
        model.fit(X_train, y_train, X_val, y_val, args.epochs)
        return model, model.history
    elif args.architecture == 'diff2vec':
        return model, None  # it is already fitted
    else:
        raise ValueError(f"Incorrect architecture: {args.architecture}")


def get_embeddings_batched(model, x, batch_size=16):
    """
    Returns the embeddings of the input data.

    Args:
        x: The input data.
        batch_size: The batch size to use for prediction. If None, uses the default batch size of the model.

    Returns:
        The embeddings of the input data.
    """

    # Create a data loader for the input data
    data_loader = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)

    # Get the embeddings using the predict method of the encoder model
    embeddings_list = []
    for batch in data_loader:
        embeddings = model.encoder.predict(batch, batch_size=batch_size)
        embeddings_list.append(embeddings)

    embeddings = np.concatenate(embeddings_list, axis=0)

    return embeddings


def get_embeddings(args, model, X_train, X_val, X_test, i_train, i_val, i_test):
    '''
    Retrieves embeddings from a trained machine learning model based on the specified architecture.

    Arguments:
    * `args` - command line arguments that specify the model configuration
    * `model` - the trained machine learning model
    * `X_train` - the input training data
    * `X_val` - the input validation data
    * `X_test` - the input testing data
    * `i_train` - the indices of training data
    * `i_val` - the indices of validation data
    * `i_test` - the indices of testing data

    Returns:
    * `E_train` - the embeddings of the training data
    * `E_val` - the embeddings of the validation data
    * `E_test` - the embeddings of the testing data
    '''
    if args.architecture == 'AE':
        E_train = get_embeddings_batched(model, X_train)
        E_val = get_embeddings_batched(model, X_val)
        E_test = get_embeddings_batched(model, X_test)
        return E_train, E_val, E_test
    elif args.architecture == 'AE_with_DLM':
        E_train = get_embeddings_batched(model.model, X_train)
        E_val = get_embeddings_batched(model.model, X_val)
        E_test = get_embeddings_batched(model.model, X_test)
        return E_train, E_val, E_test
    elif args.architecture == 'diff2vec':
        E_train = model.wv[i_train.flatten()]
        E_val = model.wv[i_val.flatten()]
        E_test = model.wv[i_test.flatten()]
        return E_train, E_val, E_test
    else:
        raise ValueError(f"Incorrect architecture: {args.architecture}")


def train_and_evaluate_classifiers(E_train, E_val, E_test, y_train, y_val, y_test):
    '''
    Trains and evaluates classifiers on the provided input embeddings and target labels.

    Arguments:
    * `E_train`: The training embeddings.
    * `E_val`: The validation embeddings.
    * `E_test`: The test embeddings.
    * `y_train`: The training target labels.
    * `y_val`: The validation target labels.
    * `y_test`: The test target labels.

    Returns:
    * A list of tuples, where each tuple contains the name of the classifier, the name of the dataset (train/val/test), 
    and the accuracy score of the classifier on that dataset.
    '''
    classifiers = {
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'LogisticRegression': LogisticRegression
    }
    accuracies = []
    for clf_name, clf in classifiers.items():
        model = clf()

        model.fit(E_train, y_train)
        for set_name, e, y in zip(['train', 'val', 'test'], [E_train, E_val, E_test], [y_train, y_val, y_test]):
            accuracies.append((
                clf_name,
                set_name,
                model.score(e, y),
            ))
    return accuracies


def get_experiment_name(args, now):
    """
    Returns a fitting experiment name with meaningful information from the input arguments.

    Args:
        args: The object generated by the argument parser.

    Returns:
        A string representing a fitting experiment name.
    """
    # Define short strings to represent each argument value
    dataset_str = args.dataset[0].upper()
    architecture_str = args.architecture
    n_blocks_str = str(args.n_blocks)
    num_filters_str = str(args.num_filters)
    pooling_str = 'P' if args.use_maxpooling else ''
    upsampling_str = 'U' if args.use_upsampling else ''
    window_size_str = str(args.sliding_window_size)
    nodes_number_str = str(args.diff_graph_nodes_number)
    labels_str = 'T' if args.use_true_labels else 'F'
    latent_dim_str = str(args.latent_dim)

    # Combine the short strings to form the experiment name
    experiment_name = f'{now}_{dataset_str}_{architecture_str}_{n_blocks_str}_{num_filters_str}_{pooling_str}_{upsampling_str}_{window_size_str}_{nodes_number_str}_{labels_str}_{latent_dim_str}'

    return experiment_name


def save_accuracies(args, exp_name, accuracies):
    '''
    Saves the accuracies of the classifiers to a CSV file.

    Arguments:
    * `args`: The command line arguments.
    * `exp_name`: The name of the experiment.
    * `accuracies`: A list of tuples containing the classifier name, the dataset name, and the accuracy score.

    Returns:
    * None.
    '''
    os.makedirs(args.summary_path, exist_ok=True)
    acc_path = os.path.join(args.summary_path, f'{exp_name}_acc.csv')
    pd.DataFrame(accuracies, columns=['classifier', 'set', 'accuracy']).to_csv(acc_path, index=False)


def save_history(args, exp_name, history):
    """
    Saves the training history to a CSV file and plots the training and validation loss curves.

    Args:
        * args: The object generated by the argument parser.
        * exp_name: A string representing the experiment name.
        * history: A dictionary containing the training history.

    Returns:
        None.
    """

    # Plot the training and validation loss curves
    losses = [loss.replace('val_', '') for loss in history.keys() if 'val_' in loss]
    n_losses = len(losses)
    fig, axs = plt.subplots(nrows=n_losses, ncols=1, figsize=(9, 6 * n_losses), dpi=300)

    if n_losses == 1:
        axs = [axs]

    for i, loss in enumerate(losses):
        axs[i].plot(history[loss], label='Training')
        if f'val_{loss}' in history:
            axs[i].plot(history[f'val_{loss}'], label='Validation')
        axs[i].set_title(f'{loss.capitalize()}')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        axs[i].legend()

    loss_curve_path = os.path.join(args.summary_path, f'{exp_name}_loss_curve.svg')
    fig.savefig(loss_curve_path)

    plt.close(fig)


def save_embeddings(args, exp_name, E_train, E_val, E_test, y_train, y_val, y_test):
    """
    Saves the embeddings of the train, validation, and test sets as scatter plots with different colors for each class
    in a single SVG file with multiple subplots.

    Args:
        args: The object generated by the argument parser.
        E_train: The embeddings of the training set.
        E_val: The embeddings of the validation set.
        E_test: The embeddings of the test set.
        y_train: The labels of the training set.
        y_val: The labels of the validation set.
        y_test: The labels of the test set.

    Returns:
        None.
    """
    # Create a PCA model and fit it to the training set embeddings
    pca = PCA(n_components=2)
    pca.fit(E_train)

    # Define a dictionary to store the embeddings and labels of each set
    embeddings_dict = {'Train': (E_train, y_train), 'Validation': (E_val, y_val), 'Test': (E_test, y_test)}

    # Create a figure with subplots for each set
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(29, 6), dpi=300)

    # Loop through each set and create a scatter plot of the embeddings with different colors for each class
    for i, (set_name, (E, y)) in enumerate(embeddings_dict.items()):
        # Apply the trained PCA model to the embeddings to reduce the dimensionality to 2
        E_2d = pca.transform(E)

        # Create a scatter plot of the embeddings with different colors for each class
        axs[i].scatter(E_2d[:, 0], E_2d[:, 1], c=plt.cm.Set1(y.astype(np.int32)))

        axs[i].set_title(f'{set_name} Set Embedding Scatter Plot')
        axs[i].set_xlabel('PC 1')
        axs[i].set_ylabel('PC 2')

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.3)

    # Save the scatter plot as a vector graphics file
    embedding_plot_path = os.path.join(args.summary_path, f'{exp_name}_embeddings.svg')
    fig.savefig(embedding_plot_path, bbox_inches='tight')

    plt.close(fig)


def summarize(args, history, now, E_train, E_val, E_test, y_train, y_val, y_test):
    '''
    Summarizes the experiment by training and evaluating classifiers, saving accuracies and embeddings.

    Arguments:
        * `args`: command line arguments that specify the experiment configuration.
        * `history`: the training history of the model.
        * `now`: the current time in datetime format.
        * `E_train`: The training embeddings.
        * `E_val`: The validation embeddings.
        * `E_test`: The test embeddings.
        * `y_train`: The training target labels.
        * `y_val`: The validation target labels.
        * `y_test`: The test target labels.

    Returns:
    None.
    '''
    exp_name = get_experiment_name(args, now)
    accuracies = train_and_evaluate_classifiers(E_train, E_val, E_test, y_train, y_val, y_test)
    save_accuracies(args, exp_name, accuracies)

    if history:
        save_history(args, exp_name, history)

    save_embeddings(args, exp_name, E_train, E_val, E_test, y_train, y_val, y_test)


def main():
    '''main function for training and testing'''
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()
    print('Reading dataset')
    G, y = return_dataset(args)
    print('Calculating features')
    features, sequences = return_features(G, args)
    X_train, X_val, X_test, y_train, y_val, y_test, i_train, i_val, i_test = train_val_test_split(features, y,
                                                                                                  np.arange(
                                                                                                      len(features)).reshape(
                                                                                                      -1, 1))
    print('Preprocessing features')
    X_train, X_val, X_test = preprocess_features(X_train, X_val, X_test, args=args)

    model = return_model(args, x_shape=X_train.shape[1:], all_classes=np.unique(y_train), sequences=sequences, now=now)
    print('Fitting model')
    model, history = fit_model(args, model, X_train, X_val, y_train, y_val)
    print('Calculating embeddings')
    E_train, E_val, E_test = get_embeddings(args, model, X_train, X_val, X_test, i_train, i_val, i_test)
    print(f'Saving summary to {args.summary_path}')
    summarize(args, history, now, E_train, E_val, E_test, y_train, y_val, y_test)


if __name__ == '__main__':
    main()
