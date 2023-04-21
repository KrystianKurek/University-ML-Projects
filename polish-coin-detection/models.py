from numba import njit, jit, prange
from numba.typed import Dict
import numba
import glob
import os
import pandas as pd
import PIL 
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np
import yaml
import tqdm 
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

class AdaBoost: 
    def __init__(self, n_iters=100, type_='stump', tree_max_depth=3): 
        self.n_iters = n_iters
        self.estimators = []
        self.b_ks = []
        self.errors = []
        self.test_errors = []
        if type_ == 'stump':
            self.clf = lambda : Tree(max_depth=tree_max_depth)
        else: 
            self.clf = lambda : DecisionTreeClassifier(max_depth=tree_max_depth, splitter='best')
        self.weight_history = []
        self.accuracy_history = []
        
    def one_hot(self, x):
        b = np.zeros((x.size, int(self.max_size+1)))
        b[np.arange(x.size),x.astype(np.int32)] = 1
        return b
    
    def fit(self, x, y): 
        self.max_size = len(set(y))

        weights = np.ones(y.shape)/len(x)
        for n_iter in tqdm.tqdm(range(self.n_iters)): 
            self.weight_history.append(weights)
            self.estimators.append(self.clf()\
                                   .fit(x, y, sample_weight=weights))
            y_pred = self.estimators[-1].predict(x)
            mask = y_pred != y
            e_k = ((mask).astype(np.int32) * weights).sum()
            b_k = e_k/(1-e_k)
            self.b_ks.append(b_k)

            
            weights[~mask] = weights[~mask]*b_k
            weights = weights/weights.sum()
            self.accuracy_history.append(self.score(x ,y))
            if (e_k == 0 or e_k >= (1 - 1/self.max_size)): 
                warnings.warn(f"e_k={e_k}, stopping AdaBoost on interation {n_iter} of {self.n_iters}")
                break
        return self 
    
    def predict(self, x): 
        y_preds = [self.one_hot(estimator.predict(x))*b_k for estimator, b_k in zip(self.estimators, self.b_ks)]
        y_preds = sum(y_preds)
        return np.argmax(y_preds, axis=1)
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)
    
    def load_best(self): 
        if self.accuracy_history:
            best_epoch = np.argmax(self.accuracy_history)
            self.estimators = self.estimators[:best_epoch+1]
            self.b_ks = self.b_ks[:best_epoch+1]
            print(f'Loaded best epoch: {best_epoch}')
            
            
class Tree: 
    def __init__(self, max_depth=5, min_sample_leaf=1): 
        possible_nodes = 2**(max_depth+1)-1
        self.best_cols = np.empty(possible_nodes)
        self.best_splits =  np.empty(possible_nodes)
        self.classes = np.empty(possible_nodes)
        self.impurities = np.empty(possible_nodes)
        self.n_samples = np.empty(possible_nodes)
        for array in [self.best_cols, self.best_splits, self.classes, self.impurities, self.n_samples]:
            array[:] = np.nan
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
    
    
    def inner_fit(self, X, y, sample_weight, depth, node_id): 
        self.impurities[node_id] = gini(y, sample_weight)
        self.n_samples[node_id] = len(X)
        
        if depth == self.max_depth or len(X) <= self.min_sample_leaf or len(np.unique(y)) == 1:
            self.classes[node_id] = most_frequent(y, sample_weight)
        else: 
            best_col, best_split, left_class, right_class = find_best_split(X, y, sample_weight)
            self.best_cols[node_id] = best_col
            self.best_splits[node_id] = best_split
            
            mask = X[:, best_col] <= best_split
            left_y, right_y = y[mask], y[~mask]
            left_sample_weight, right_sample_weight = sample_weight[mask], sample_weight[~mask]
            left_x, right_x = X[mask], X[~mask]
            
            self.inner_fit(left_x, left_y, left_sample_weight, depth+1, 2*node_id+1)
            
            self.inner_fit(right_x, right_y, right_sample_weight, depth+1, 2*node_id+2)

                
        
    def fit(self, X, y, sample_weight):
        self.inner_fit(X, y, sample_weight, 0, 0)
        return self 
    
    def inner_predict(self, x, node_id):
        if not np.isnan(self.best_splits[node_id]):
            best_split, best_col = self.best_splits[node_id], self.best_cols[node_id]
            mask = x[:, int(best_col)] <= best_split
            y_pred_left = self.inner_predict(x[mask], 2*node_id+1)
            y_pred_right = self.inner_predict(x[~mask], 2*node_id+2)
            y_pred = np.zeros(len(x))
            y_pred[mask] = y_pred_left
            y_pred[~mask] = y_pred_right
            return y_pred
        else: 
            y_pred = np.zeros(len(x)) + self.classes[node_id]
            return y_pred
    
    def predict(self, x): 
        return self.inner_predict(x, node_id=0)
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)

@njit(parallel=True)
def find_best_split(X, y, weights):
    y = y.astype(np.float32)
    indices = np.arange(X.shape[1])
    np.random.permutation(indices)
    best_global_impurity = np.empty(X.shape[1])
    best_global_splits = np.empty(X.shape[1])
    for idx in prange(X.shape[1]): 
        i = indices[idx]
        col = X[:, i].flatten()
        unique_col = np.sort(np.unique(col))
        best_sub_impurity = np.empty(unique_col.shape[0])
        for split_idx in prange(unique_col.shape[0]): 
            split = unique_col[split_idx]
            mask = col <= split
            left_impurity = gini(y[mask], weights[mask])
            right_impurity = gini(y[~mask], weights[~mask])
            impurity = mask.sum()*left_impurity + right_impurity*(~mask).sum()
            impurity = impurity/len(mask)

            best_sub_impurity[split_idx] = impurity

        argmin_imp = np.argmin(best_sub_impurity)
        best_global_impurity[i] = best_sub_impurity[argmin_imp]
        best_global_splits[i] = unique_col[argmin_imp]
    
    argmin_imp = np.argmin(best_global_impurity)
    best_global_split = best_global_splits[argmin_imp]
    mask = X[:, argmin_imp] <= best_global_split
    left_class = most_frequent(y[mask], weights[mask])
    right_class = most_frequent(y[~mask], weights[~mask])
    return argmin_imp, best_global_split, left_class, right_class
    

@njit(parallel=True)
def gini(samples, weights):
    if len(samples) == 0:
        return 0
    unique_samples = np.unique(samples)
    ginis = np.empty(len(unique_samples))
    for i in prange(len(unique_samples)):
        mask = samples == unique_samples[i]
        p = np.sum((mask).astype(np.int32) * weights) / weights.sum()
        ginis[i] = p*(1 - p)
    return ginis.sum()

@njit
def most_frequent(labels, sample_weights):
    counts = numba.typed.Dict.empty(key_type=numba.float32, value_type=numba.float64)
    for i in range(len(labels)):
        x = labels[i]
        if x in counts:
            counts[x] += sample_weights[i]
        else:
            counts[x] = sample_weights[i]
    most_common = -1
    max_count = -1
    for key, value in counts.items():
        if value > max_count:
            most_common = key
            max_count = value
    return most_common

class NearestCentroid: 
    def __init__(self): 
        pass
    
    def fit(self, x, y): 
        self.means = []
        for class_ in set(y): 
            self.means.append(x[y == class_].mean(axis=0))
        return self 
    
    def predict(self, x):
        return [np.argmin([np.linalg.norm(mean - x_elem) for mean in self.means]) for x_elem in x]
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)
        
class PCA: 
    def __init__(self):
        self.fitted = False
        
    def fit(self, X): 
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        X = (X - self.X_mean)/self.X_std
        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        self.sort_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues#[self.sort_indices]
        self.eigenvectors = eigenvectors#[:, self.sort_indices]
        self.X_mean = X.mean(axis=0)
        self.fitted = True
        
    def transform(self, X, n_components=10): 
        if not self.fitted: 
            raise ValueError("You have to fit first!")
        X = (X - self.X_mean)/self.X_std
        return (self.eigenvectors[:, self.sort_indices][:, :n_components].T @ (X).T).T
    
    def fit_transform(self, X, n_components=10):
        self.fit(X)
        return self.transform(X, n_components=n_components)
    
    def return_eigenvalues(self):
        return self.eigenvalues