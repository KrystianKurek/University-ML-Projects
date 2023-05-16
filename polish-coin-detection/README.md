For this project we collected small [dataset](https://drive.google.com/drive/folders/1cyy5ceKFiMETvp3OmuazN2kK8WG6_BO0) 
with images of Polish coins. Since each image contains multiple coins,  we manually indicated the bounding boxes and nominal values for each coin visible on the obverse side.

My friend took care of detecting the coins, while I focused on 
implementing feature 
extraction and classification methods, including:
* PCA - Principal Component Analysis for features extractiona and dimensionality reduction
* Decision trees, AdaBoosting and Nearest Centroid classifiers

As it was required to implement classical methods of feature extraction and classification, we 
couldn't use deep learning methods such as YOLO.

To see an example of working detection check `demo.ipynb` notebook.
