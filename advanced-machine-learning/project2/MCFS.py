import asyncio
import nest_asyncio
nest_asyncio.apply()
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import chi2
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def count_gain_ratio_and_feature(tree, node_id):
    left_id = tree.children_left[node_id]
    right_id = tree.children_right[node_id]
    if left_id == right_id == -1: 
        return tree.feature[node_id], 0
    

    n_samples = tree.n_node_samples[node_id]
    n_samples_left = tree.n_node_samples[left_id]
    n_samples_right = tree.n_node_samples[right_id]

    return tree.feature[node_id], tree.impurity[node_id] - (n_samples_left*tree.impurity[left_id] + n_samples_right*tree.impurity[right_id])/n_samples
#count_gain_ratio_and_feature(dtc.tree_, 1)


async def return_single_tree_feature_importance(x_cut_features, y, u, v):
    x_train, x_test, y_train, y_test = return_random_split(x_cut_features, y)
    dtc = DecisionTreeClassifier(max_depth=3, criterion='gini').fit(x_train, y_train)
    single_tree_feature_importance = np.zeros(shape=(x_cut_features.shape[1], 1))
    for node_id in range(dtc.tree_.node_count): 
        feature_id, gain_ratio = count_gain_ratio_and_feature(dtc.tree_, node_id)
        single_tree_feature_importance[feature_id] += gain_ratio*dtc.tree_.n_node_samples[node_id]
    single_tree_feature_importance = (single_tree_feature_importance/len(x_test))**v
    accuracy = accuracy_score(dtc.predict(x_test), y_test)**u
    single_tree_feature_importance *= accuracy
    return single_tree_feature_importance.reshape(-1, 1)

def return_feature_importance_for_x_cut(x_cut_features, y, u, v, t):
    loop = asyncio.get_event_loop()
    tasks = [return_single_tree_feature_importance(x_cut_features, y, u, v)]*t
    results = loop.run_until_complete(asyncio.gather(*tasks))
    results = np.concatenate(results, axis=1)
    
    results = results.sum(axis=1)
    return results
 
def return_random_split(x, y):
    x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.333)
    return x_train, x_test, y_train, y_test

def return_MCFS(x, y, m=None ,t = 1000,s = 500,v = 1.0,u = 1.0):
    if m is None: 
        m = x.shape[1] // 10
    feature_importance = np.zeros(shape=(x.shape[1], ))
    for _ in range(s):
        indices = np.random.choice(np.arange(x.shape[1]), replace=False, size=x.shape[1] - m)
        x_cut_features = np.copy(x)
        x_cut_features[:, indices] = 0.0
            

        feature_importance += return_feature_importance_for_x_cut(x_cut_features, y, u, v, t)
    return feature_importance

        