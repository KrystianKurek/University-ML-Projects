Together with my [friend](https://github.com/gozderamichal) we've implemented [diff2vec](https://arxiv.org/abs/2001.07463) (for extracting node embeddings) algorithm and tried to enhance it. 
We later used embeddings for classification task. 
He focused mainly on implementing random walks on diffusion graphs using the Numba library, 
which significantly increased the speed and performance of the algorithm.

In my work, I primarily concentrated on using frequency vectors extracted from random walks with 
autoencoders. We used convolutional autoencoders to recreate frequency vectors for each node, 
encoding them in a lower-dimensional space that served as our embeddings.

Additionally, we enhanced the embeddings by using contrastive loss, which made nodes' embeddings 
from the same class close 
together in the lower-dimensional space and far apart when they were from different classes. 

We evaluated the original and enhanced versions of the Diff2Vec algorithm using three datasets:
* An artificial graph with 1000 nodes and 4 communities
* The Twitch graph with almost 8k nodes and 2 classes
* The LastFM graph with over 7k nodes and 19 classes

We found that the original Diff2Vec algorithm (based on word2vec) performed better 
in terms of speed and classification performance than our enhancement. However, our enhanced 
algorithm performed similarly to the 
original version in the case of the artificial graph and was not far behind for larger graphs.


------

The code structure is organized as follows:

* The source code is located in the `src` folder.
* The obtained results can be found in `summary` folder.
* `src/data_utils` contains functions for loading real graph data.
* `src/diff2vec` contains code used for feature extraction parts (the implementation in Python and `networkx` is available in `src/diff2vec`, and the version implemented in `numpy` and `numba` is in `src/diff2vec/numba`).
* `src/models` contains autoencoder models implementations.
* `src/test_utils` contains functions to generate artificial graph data.
* `src/main.py` can be used to train and test the autoencoder models.
* `src/numba_diffuser_tests.ipynb`, `src/numba_euler_path_tests.ipynb`, `src/numba_feature_extraction_tests.ipynb` notebooks contain time performance tests for comparing different feature extraction methods (respectively for creating diffusion graphs, finding Euler's walk, the entire feature extraction process)

## Instalation 
```
conda create --name env_name python=3.7.6
conda activate env_name
conda install --file src/requirements_conda.txt
pip install -r src/requirements_pip.txt
```