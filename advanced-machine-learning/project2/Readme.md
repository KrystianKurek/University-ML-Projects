My classmate and I used various methods for feature selection, including:

* Chi-squared test
* Boruta
* Monte Carlo Feature Selection (which we implemented ourselves, but with a small yet important error - can you spot it?)
* RandomForest feature selection from the sklearn library

We combined these methods with two classifiers, XGBoost and Random Forest, and evaluated their performance using 5-fold cross-validation.